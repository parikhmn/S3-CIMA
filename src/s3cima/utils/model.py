"""Utils specific to the S3-CIMA Model classes.

Overview of the training pipeline (``fit``)
-------------------------------------------
1. Split patients into a held-out test set (1 patient per condition) and
   ``n_val_folds`` train/validation folds (see utils/datasets.py).

2. Train ``nruns`` independent runs. Each run draws its own hyper-parameters:
     - number of filters  : random draw from ``nfilter_selection``
     - pooling size k     : cycles through ``maxpool_percentages`` (as % of ncell)
   Within a run, a FRESH model is trained on every fold, and one checkpoint
   per run is kept: the fold/epoch with the best validation balanced
   accuracy ([FIX-1]).

   A filter is NOT a subset of markers. Every filter is a weight vector over
   ALL markers (plus a bias), i.e. a learned "cell signature". The random
   choice is HOW MANY signatures a model may use, not which markers it sees.

3. Consensus filters (``get_discriminative_filters``):
     - every saved run is re-scored on the validation fold it was selected on
       ([FIX-2]); runs at or above chance-level ((C-1)/C) are kept;
     - filters that mainly vote for the background class are dropped;
     - all remaining filters from all kept runs are pooled and clustered by
       cosine distance; the medoid of each cluster is its consensus filter.
       Clusters with many members from different runs are signatures that
       several independently trained models converged on;
     - ``filter_diff`` (a simple activation difference between the filter's
       favoured class and the other classes) flags discriminative filters.

4. Test ranking: the models that produced each discriminative consensus
   filter are scored on the held-out test patients, and the top 4 filters
   by test accuracy are passed on for plotting. See the comment block at
   that step for how to interpret this.

Notes on fixes
--------------
Changes to the original code are marked with ``[FIX-n]`` blocks. The original
line(s) are kept commented out directly above the replacement. See FIXES.md
at the repository root for a summary.
"""

# Imports

import os
import argparse
import pickle
import random
import time
import json
from datetime import datetime

from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, cdist

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from sklearn.metrics import f1_score, balanced_accuracy_score

from s3cima.utils.datasets import CIMADataset
from s3cima.utils.datasets import spatial_collate_fn, get_norm_stats
from s3cima.utils.datasets import make_patient_stratified_splits

from s3cima.model import CellCNN

# General functions ---------------------------------------------------------------

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.

    Tracks the best validation loss seen so far; if it fails to improve by
    more than ``min_delta`` for ``patience`` consecutive epochs,
    ``early_stop`` is set to True and the training loop breaks.
    """
    def __init__(self, patience=10, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            # First epoch: initialise the reference loss
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            # Loss improved: new reference value
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            # No improvement: count towards patience
            self.counter += 1
            print(f"[INFO]: Early stopping counter - {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('[INFO]: Early stopping')
                self.early_stop = True


def set_seed(seed = 42):
    """Fix the seeds for reproductible runs during training"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


# Filter calculation functions ----------------------------------------------------

def _relu(x):
    return np.maximum(0, x)


def _single_filter_output(conv_w, bias, out_w, valid_samples, mp):
    """
    Score every neighbourhood with ONE filter, outside of the full model.

    For each neighbourhood:
        cell response  g_c = ReLU(x_c · w + bias)         for each cell c
        pooled value       = mean of the top `ntop` g_c    (ntop = mp% of cells)

    The filter's "dominant class" is the class its output-layer weight
    points to most strongly, i.e. which condition a high response votes for.

    conv_w  : (nmark,)   — filter weights
    bias    : float      — filter bias
    out_w   : (n_classes,) — output weights for this filter
    valid_samples : list of (ncell, nmark) arrays — should be z-scored the
                    same way as during training (see [FIX-2])
    mp      : float      — max-pooling percentage (0-100)

    Returns
    -------
    y_pred         : (n_samples,) pooled filter response per neighbourhood
    dominant_class : int
    """
    y_pred = np.zeros(len(valid_samples))
    for i, x in enumerate(valid_samples):
        # x shape: (ncell, nmark)
        g = _relu(x @ conv_w + bias)               # (ncell,)
        ntop = max(1, int(mp / 100. * x.shape[0]))
        gpool = np.mean(np.sort(g)[-ntop:])            # scalar
        y_pred[i] = gpool
    dominant_class = int(np.argmax(out_w))
    return y_pred, dominant_class


def _compute_filter_diff(consensus_conv, consensus_out, consensus_biases,
                          valid_samples, valid_phenotypes, mp):
    """
    Returns filter_diff array of shape (n_consensus_filters,).

    filter_diff = mean pooled response on neighbourhoods of the filter's
                  dominant class  −  mean pooled response on all other
                  neighbourhoods.

    Interpretation:
      - > 0 : the filter responds more in the class it votes for (good);
      - it is a raw difference in activation units, with no significance
        test, and its scale depends on the filter's weight magnitude, so it
        is not strictly comparable between filters.
      - pooling is fixed by the caller (currently mp = 100, i.e. averaging
        over all cells) regardless of the k the source model was trained with.
    """
    y_true = np.array(valid_phenotypes)
    filter_diff = np.zeros(len(consensus_conv))

    for i, (conv_w, out_w, bias) in enumerate(
            zip(consensus_conv, consensus_out, consensus_biases)):
        y_pred, filter_class = _single_filter_output(
            conv_w, bias, out_w, valid_samples, mp
        )
        in_class = y_true == filter_class
        out_class = ~in_class
        if in_class.sum() == 0 or out_class.sum() == 0:
            filter_diff[i] = 0.0   # degenerate case
        else:
            filter_diff[i] = np.mean(y_pred[in_class]) - np.mean(y_pred[out_class])

    return filter_diff


def get_discriminative_filters(
    save_model: str,
    nruns: int,
    # [FIX-2] The single, un-normalised `validation_subset` argument is
    # replaced by the list of normalised folds, so each run can be scored on
    # its own validation fold. Original parameter (kept for reference):
    # validation_subset: Subset,
    folds: list,
    loss_fn,
    n_classes: int,
    nmarkers: int,
    lr: float,
    batch_size: int,
    num_workers: int,
    log_file: str,
    device: str,
    accur_thres: float = 0.85,
    dendrogram_cutoff: float = 0.4,
    filter_diff_threshold: float = 0.0,
    result_folder: str = None,
    genes: list = None,
    has_background: bool = True,
    classification: bool = True,
):
    """
    After all runs are complete:
      1. Load each saved run's best model
      2. Evaluate validation accuracy on the validation fold that checkpoint
         was selected on (z-scored as in training) — see [FIX-2]
      3. Collect filter weight vectors from runs passing accur_thres
         (if no run passes, ALL runs are kept)
      4. Concatenate all passing filter weights into one matrix
      5. Cluster via hierarchical clustering (cosine distance)
      6. Select one consensus filter per cluster (medoid) and calculate filter diff

    Filter weights shape per run: (nfilter, nmark)
    Only filters connected to non-control classes are retained
    (i.e. output weight to ctrl class [n_classes-1] is NOT the dominant connection).

    Parameters
    ----------
    folds : list of (train NormalizedCIMASubset, val NormalizedCIMASubset)
            as returned by make_patient_stratified_splits.

    Outputs written to <result_folder>/filters/
    -------------------------------------------
    consensus_filter_weights.npy   (n_clusters, nmark) medoid weights
    consensus_output_weights.npy   (n_clusters, n_classes)
    all_pooled_*.npy               every pooled filter before clustering
    cluster<id>_filter_weights.csv ranked marker weights per consensus filter
    consensus_filter_meta.json     per-cluster metadata (see below)
    filter_discriminability.csv    filter_diff per cluster

    Key metadata fields
    -------------------
    n_members   : cluster size. >1 members from DIFFERENT runs means several
                  independently trained models converged on this signature —
                  the most useful robustness signal.
    source_run  : run that produced the medoid filter.
    source_fold : validation fold the source run's checkpoint was selected on.
    val_ba      : validation balanced accuracy of the source RUN (a property
                  of the whole model, not of the individual filter).
    """
    # Block if regression for the moment
    if not classification:
        raise Exception("Regression not implemented yet")

    # Validate genes if provided
    if genes is not None:
        genes = list(genes)
        if len(genes) != nmarkers:
            raise ValueError(
                f"Length of genes ({len(genes)}) must equal nmarkers ({nmarkers})."
            )

    # ------------------------------------------------------------------
    # [FIX-2] Score each run on normalised data from its own fold
    # ------------------------------------------------------------------
    # Problem 1 (normalisation): fit() passed `validation_dataset.subset`,
    #   i.e. the raw Subset underneath the NormalizedCIMASubset. Models
    #   trained on z-scored inputs were therefore re-scored (and filter_diff
    #   computed) on RAW intensities they had never seen. Both the run filter
    #   (accur_thres) and filter_diff rested on these mismatched inputs.
    #
    # Problem 2 (fold reuse): a single fold was chosen at random for all
    #   runs. A checkpoint saved from fold A has trained on the validation
    #   patients of fold B, so scoring it on fold B is scoring it on its own
    #   training data (optimistic, and different for each run).
    #
    # Fix: each checkpoint now records the fold it came from ("fold" key, see
    #   [FIX-1] in fit()). Every run is scored on THAT fold's normalised
    #   validation subset, and each consensus filter's filter_diff is computed
    #   on the validation fold of its source run. Older checkpoints without a
    #   "fold" key fall back to fold 0 with a warning.
    #
    # Original code (kept for reference):
    # # Get the sample information from the validation dataset
    # t_loader = DataLoader(
    #     validation_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    # )
    def _val_loader(fold_idx):
        return DataLoader(
            folds[fold_idx][1], batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
        )
    run_folds = {}        # run -> index of the fold its checkpoint was selected on
    # ------------------------------------------------------------------

    accuracies = {}
    f1_scores = {}
    all_weights = {}      # run -> dict with 'conv' (nfilter, nmark) and 'out' (nfilter, n_classes)

    # ----------------------------------------------------------------
    # 1. Load each run, evaluate validation accuracy, extract weights
    # ----------------------------------------------------------------
    for run in range(nruns):
        ckpt_path = f"{save_model}/best_model_run{run}.pth"
        if not os.path.exists(ckpt_path):
            print(f"[WARN] No checkpoint for run {run}, skipping.")
            continue

        params = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Rebuild the exact architecture saved in the checkpoint
        model = CellCNN(
            nmark = params["nmarkers"],
            nfilter = params["nfilter"],
            k = params["k"],
            n_classes = params["n_classes"],
            dropout = params["dropout"],
            dropout_p = params["dropout_p"],
            regression = params["regression"],
            selection_type = params["selection_type"],
        )
        model.load_state_dict(params["model_state_dict"])
        model.to(device)

        # [FIX-2] pick the run's own (normalised) validation fold
        fold_idx = params.get("fold")
        if fold_idx is None:
            print(f"[WARN] Checkpoint for run {run} has no 'fold' entry "
                  f"(older format); scoring it on fold 0.")
            fold_idx = 0
        run_folds[run] = fold_idx
        t_loader = _val_loader(fold_idx)

        test_loss, preds, trues = test(
            model, t_loader, loss_fn, 0, result_folder, log_file, device
        )

        ba = balanced_accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average = "macro")
        accuracies[run] = ba
        f1_scores[run] = f1
        print(f"[Run {run}] Validation balanced accuracy (fold {fold_idx}): {ba:.4f}")
        print(f"[Run {run}] Validation F1 Score: {f1:.4f}")

        # Extract conv filter weights: shape (nfilter, nmark)
        # Extract output layer weights: shape (nfilter, n_classes)
        # Careful of cellcnn names
        conv_w = model.conv.weight.detach().cpu().numpy()   # (nfilter, nmark, 1) for Conv1d
        conv_w = conv_w.squeeze(-1)                          # (nfilter, nmark)
        out_w  = model.fc.weight.detach().cpu().numpy()  # (n_classes, nfilter)
        out_w  = out_w.T                                     # (nfilter, n_classes)
        bias_w = model.conv.bias.detach().cpu().numpy()

        all_weights[run] = {"conv": conv_w, "out": out_w, "bias": bias_w}

        with open(log_file, "a") as log:
            log.write(f"\n[Run {run}] Validation balanced accuracy (fold {fold_idx}): {ba:.2f}, Validation Macro F1 Score: {f1:.2f}\n")

    # Select runs above threshold. fit() sets accur_thres = (C-1)/C, which for
    # a 2-class problem is 0.5 (chance), so in practice most runs pass.
    sorted_runs = sorted(accuracies, key=accuracies.get, reverse=True)
    passing_runs = [r for r in sorted_runs if accuracies[r] >= accur_thres]

    default_selection = len(sorted_runs)
    if len(passing_runs) < 1:
        # Fallback: no run passed, so ALL runs are kept.
        passing_runs = sorted_runs[:default_selection]
        # [FIX-6] The message said "top 3", but default_selection keeps every
        # run. Original line (kept for reference):
        # print(f"[INFO] No run passed the accur_thres={accur_thres}. "
        #       f"Keeping top 3 by accuracy.")
        print(f"[INFO] No run passed the accur_thres={accur_thres}. "
              f"Keeping all {default_selection} runs.")

        with open(log_file, "a") as log:
            log.write(f"[INFO] No run passed the accur_thres={accur_thres}.\n")
            log.write(f"[INFO] Keeping top {default_selection} runs by accuracy.\n")

    print(f"[INFO] Runs used for filter extraction: {passing_runs}")

    # ----------------------------------------------------------------
    # 3. Pool filter weight vectors from passing runs
    #    Keep only filters whose dominant output connection is NOT
    #    the control class
    # ----------------------------------------------------------------
    # The background class is always encoded last (index n_classes - 1).
    # This MUST only be enabled when a background class really exists,
    # otherwise the last REAL condition is mistaken for background ([FIX-4]).
    ctrl_class = n_classes - 1 if has_background else None

    pooled_conv = []
    pooled_out = []
    pooled_bias = []
    filter_run_ids = []

    for run in passing_runs:
        conv_w = all_weights[run]["conv"]   # (nfilter, nmark)
        out_w  = all_weights[run]["out"]    # (nfilter, n_classes)
        bias_w = all_weights[run]["bias"]   # (nfilter,)

        for f_idx in range(conv_w.shape[0]):
            if has_background:
                if np.argmax(out_w[f_idx]) == ctrl_class:
                    continue # discard the filters pertaining to the background
            pooled_conv.append(conv_w[f_idx])
            pooled_out.append(out_w[f_idx])
            pooled_bias.append(bias_w[f_idx])
            filter_run_ids.append(run)

    if len(pooled_conv) == 0:
        # NOTE: fit() does not handle this None return and will fail when it
        # reads filter_results['meta'].
        print("[WARN] No non-control filters found across passing runs.")
        with open(log_file, "a") as log:
            log.write(f"[WARN] No non-control filters found across passing runs.\n")
        return None

    pooled_conv = np.array(pooled_conv)
    pooled_out  = np.array(pooled_out)

    print(f"[INFO] Total non-control filters pooled: {pooled_conv.shape[0]}")

    # ----------------------------------------------------------------
    # 4. Hierarchical clustering of filter weight vectors
    #    using cosine distance
    # ----------------------------------------------------------------
    # Cosine distance compares the PATTERN of weights across markers and
    # ignores overall magnitude. Filter numbering is arbitrary across models
    # (filter #2 of run 4 has nothing to do with filter #2 of run 7), so the
    # weights are the only thing that can be compared. Average linkage cut at
    # `dendrogram_cutoff` (default 0.4 ≈ cosine similarity > 0.6 on average).
    if pooled_conv.shape[0] == 1:
        # Edge case: only one filter, no clustering needed
        labels_clust = np.array([0])
    else:
        dist_matrix  = pdist(pooled_conv, metric="cosine")
        linkage_mat  = linkage(dist_matrix, method="average")
        labels_clust = fcluster(linkage_mat, t=dendrogram_cutoff, criterion="distance")

    n_clusters = len(np.unique(labels_clust))
    print(f"[INFO] Number of filter clusters: {n_clusters}")

    # ----------------------------------------------------------------
    # 5. Select consensus filter per cluster: the medoid
    #    (filter with minimum average cosine distance to all others
    #     in the same cluster)
    # ----------------------------------------------------------------
    consensus_conv = []
    consensus_out  = []
    consensus_meta = []
    consensus_bias = []

    for cluster_id in np.unique(labels_clust):
        member_idx = np.where(labels_clust == cluster_id)[0]
        members    = pooled_conv[member_idx]

        if len(member_idx) == 1:
            medoid_local = 0
        else:
            dist_mat = cdist(members, members, metric="cosine")
            medoid_local = np.argmin(dist_mat.mean(axis=1))

        medoid_global = member_idx[medoid_local]
        medoid_weights = pooled_conv[medoid_global]    # (nmark,)

        # Rank genes by absolute filter weight (descending)
        # Reminder: a weight only matters if its marker actually varies
        # across cells. Large weights on rarely detected markers mostly
        # reflect the random initialisation, not biology.
        ranked_idx = np.argsort(np.abs(medoid_weights))[::-1]
        ranked_weights = medoid_weights[ranked_idx].tolist()
        ranked_genes = (
            [genes[i] for i in ranked_idx] if genes is not None
            else [f"marker_{i}" for i in ranked_idx]
        )

        consensus_conv.append(medoid_weights)
        consensus_out.append(pooled_out[medoid_global])
        consensus_bias.append(pooled_bias[medoid_global])
        consensus_meta.append({
            "cluster":            int(cluster_id),
            "medoid_filter_idx":  int(medoid_global),
            "source_run":         filter_run_ids[medoid_global],
            # [FIX-2] record the fold the source run was scored on
            "source_fold":        int(run_folds[filter_run_ids[medoid_global]]),
            "val_ba":             round(accuracies[filter_run_ids[medoid_global]], 4),
            "n_members":          len(member_idx),
            "top_genes":          ranked_genes[:10],
            "top_weights":        [round(w, 6) for w in ranked_weights[:10]],
            "all_gene_weights":   dict(zip(ranked_genes, [round(w, 6) for w in ranked_weights])),
        })

    consensus_conv = np.array(consensus_conv)  # (n_clusters, nmark)
    consensus_out  = np.array(consensus_out)   # (n_clusters, n_classes)
    consensus_bias = np.array(consensus_bias)  # (n_clusters,)

    # ----------------------------------------------------------------
    # 6a. Compute filter_diff if validation data provided
    # ----------------------------------------------------------------
    filter_dir = os.path.join(result_folder, "filters")
    os.makedirs(filter_dir, exist_ok=True)

    filter_diff = None

    # [FIX-2] filter_diff was computed on RAW intensities read directly from
    # the underlying dataset (`validation_subset.dataset.samples`), and on the
    # randomly chosen fold. It is now computed on z-scored neighbourhoods
    # (read through the NormalizedCIMASubset) from the validation fold of each
    # filter's source run.
    #
    # Original code (kept for reference):
    # valid_samples = [validation_subset.dataset.samples[i] for i in validation_subset.indices]
    #
    # vs = [
    #     s["intensity"].numpy() if isinstance(s, dict) else np.array(s)
    #     for s in valid_samples
    # ]
    #
    # # Default to 100% (mean pooling) as a safe fallback
    # mp = 100.0
    # filter_diff = _compute_filter_diff(
    #     consensus_conv, consensus_out, consensus_bias, vs, trues, mp
    # )
    fold_data = {}   # fold index -> (list of z-scored (ncell, nmark) arrays, labels)

    def _fold_arrays(fold_idx):
        if fold_idx not in fold_data:
            val_subset = folds[fold_idx][1]
            items = [val_subset[i] for i in range(len(val_subset))]
            fold_data[fold_idx] = (
                [it["intensity"].numpy() for it in items],
                [int(it["label"].item()) for it in items],
            )
        return fold_data[fold_idx]

    # Default to 100% (mean pooling) as a safe fallback
    # NOTE: this ignores the pooling size k the source model was trained with.
    mp = 100.0
    filter_diff = np.zeros(len(consensus_conv))
    for i, m in enumerate(consensus_meta):
        vs, v_labels = _fold_arrays(m["source_fold"])
        filter_diff[i] = _compute_filter_diff(
            consensus_conv[i:i + 1], consensus_out[i:i + 1],
            consensus_bias[i:i + 1], vs, v_labels, mp
        )[0]

    print(f"[INFO] Filter discriminability scores: {np.round(filter_diff, 4)}")

    # Flag low-discriminability filters
    flagged = [i for i, fd in enumerate(filter_diff) if fd < filter_diff_threshold]
    if flagged:
        print(f"[WARN] Filters with filter_diff < {filter_diff_threshold}: "
                f"cluster IDs {[consensus_meta[i]['cluster'] for i in flagged]}")

    # Embed into metadata
    for i, m in enumerate(consensus_meta):
        m["filter_diff"] = round(float(filter_diff[i]), 6)
        m["discriminative"] = bool(filter_diff[i] >= filter_diff_threshold)

    # Save filter_diff as CSV
    df_diff = pd.DataFrame({
        "cluster":      [m["cluster"] for m in consensus_meta],
        "filter_diff":  filter_diff,
        "discriminative": [m["discriminative"] for m in consensus_meta],
    }).sort_values("filter_diff", ascending=False)
    df_diff.to_csv(f"{filter_dir}/filter_discriminability.csv", index=False)

    with open(log_file, "a") as log:
        log.write(f"\n[FILTER DIFF] Discriminability scores (threshold={filter_diff_threshold}):\n")
        for _, row in df_diff.iterrows():
            log.write(f"  Cluster {int(row['cluster'])}: "
                        f"filter_diff={row['filter_diff']:.4f} "
                        f"({'OK' if row['discriminative'] else 'FLAGGED'})\n")

    # ----------------------------------------------------------------
    # 7. Save everything
    # ----------------------------------------------------------------
    np.save(f"{filter_dir}/consensus_filter_weights.npy", consensus_conv)
    np.save(f"{filter_dir}/consensus_output_weights.npy", consensus_out)
    np.save(f"{filter_dir}/all_pooled_filter_weights.npy", pooled_conv)
    np.save(f"{filter_dir}/all_pooled_output_weights.npy", pooled_out)

    # Save per-cluster ranked gene weight CSVs
    col_label = "gene" if genes is not None else "marker"
    for m in consensus_meta:
        cluster_id = m["cluster"]
        df_filter = pd.DataFrame({
            col_label: list(m["all_gene_weights"].keys()),
            "weight":  list(m["all_gene_weights"].values()),
        })
        # Already sorted by |weight| descending from ranked_idx above
        df_filter.to_csv(
            f"{filter_dir}/cluster{cluster_id}_filter_weights.csv", index=False
        )

    with open(f"{filter_dir}/consensus_filter_meta.json", "w") as f:
        json.dump(consensus_meta, f, indent=2)

    with open(log_file, "a") as log:
        log.write(f"\n[FILTERS] {n_clusters} consensus filters saved to {filter_dir}\n")
        for m in consensus_meta:
            log.write(f"  Cluster {m['cluster']}: {m['n_members']} members, "
                      f"source run={m['source_run']}\n")

    print(f"[INFO] Consensus filters saved to {filter_dir}")
    return {
        "consensus_conv":   consensus_conv,
        "consensus_out":    consensus_out,
        "all_pooled_conv":  pooled_conv,
        "all_pooled_out":   pooled_out,
        "accuracies":       accuracies,
        "f1_scores":        f1_scores,
        "passing_runs":     passing_runs,
        "cluster_labels":   labels_clust,
        "meta":             consensus_meta,
    }

# Model training and fitting functions ---------------------------------------------------------

def train(model,
          train_loader,
          loss_fn,
          optimizer,
          epoch,
          log_file,
          device):
    '''Train the model for ONE epoch (one pass over all training batches).

    Each batch: forward pass -> cross-entropy loss -> backpropagation ->
    Adam step. Balanced accuracy and macro F1 are computed over the whole
    epoch (balanced accuracy is used because classes can be imbalanced,
    e.g. very different numbers of anchors per condition).

    Arguments
    ---
    model : nn.Module descendant
        Model through which training data is passed
    train_loader : DataLoader
        Training data with features and labels
    loss_fn : function
        Method of loss calculation
    optimizer : function (taken using torch.optim)
        Optimisation method
    epoch : int
        Current epoch index (used for logging)
    log_file : str
        Path of the text log
    device : str
        "cuda" or "cpu"

    Returns
    ---
    epoch_loss : float
        Mean loss over batches
    ba : float
        Balanced accuracy over the epoch
    f1 : float
        Macro F1 over the epoch
    '''
    model.train()  # enables dropout

    train_loss = 0
    counter = 0

    all_true_classes = []
    all_pred_classes = []
    with tqdm(train_loader, total=len(train_loader), unit="batch") as tepoch:
        for batch in tepoch:
            counter += 1
            tepoch.set_description(f"Epoch {epoch}")

            # Send the input to the device
            X, y = batch['intensity'].to(device), batch['label'].to(device)

            # Compute prediction and loss
            pred = model(X)

            # Loss calculation
            loss = torch.sum(loss_fn(pred, y))

            # Backpropagation
            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Batch metrics - first detach vectors
            pred = pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            # Predicted class = highest logit
            class_preds = np.argmax(pred, axis = 1)

            # Save class information
            all_true_classes.extend(y)
            all_pred_classes.extend(class_preds)

            # Progres pbar
            postfix = {}
            postfix["Train: loss"] = f"{train_loss / counter:.5f}"
            tepoch.set_postfix(postfix)

        # Loss and protein metrics
        epoch_loss = train_loss / counter
        ba = balanced_accuracy_score(all_true_classes, all_pred_classes)
        f1 = f1_score(all_true_classes, all_pred_classes,
                      average = "macro")

        # Save to log (NOTE: "Loss" logged here is the LAST batch's loss)
        with open(log_file, "a") as log:
            log.write(f"Epoch {epoch} train : Balanced Accuracy = {ba:.2f}, Macro F1-score = {f1:.2f} | Loss : {loss:.2f}\n")

    return epoch_loss, ba, f1


def validate(model, val_loader, loss_fn, epoch, log_file, device):
    '''Evaluate the model on the validation fold after an epoch (no gradient).

    Arguments
    ---
    model : nn.Module descendant
        Model through which data is passed
    val_loader : DataLoader
        Validation data with features and labels
    loss_fn : function
        Method of loss calculation
    epoch : int
        Current epoch index (used for logging)
    log_file : str
        Path of the text log
    device : str
        "cuda" or "cpu"

    Returns
    ---
    epoch_loss : float
        Mean validation loss over batches (used for early stopping)
    ba : float
        Validation balanced accuracy (used for checkpoint selection)
    f1 : float
        Validation macro F1
    '''
    model.eval()  # disables dropout

    val_loss = 0
    counter = 0

    all_true_classes = []
    all_pred_classes = []
    with torch.no_grad():
        for batch in val_loader:
            counter += 1

            #Send the input to the device
            X, y = batch['intensity'].to(device), batch['label'].to(device)

            #Compute prediction and loss
            pred = model(X)

            # Loss calculation
            loss = torch.sum(loss_fn(pred, y))
            val_loss += loss.item()

            # Batch metrics - first detach vectors
            pred = pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            class_preds = np.argmax(pred, axis = 1)

            # Save class information
            all_true_classes.extend(y)
            all_pred_classes.extend(class_preds)

        #Loss and protein metrics
        epoch_loss = val_loss / counter

        ba = balanced_accuracy_score(all_true_classes, all_pred_classes)
        f1 = f1_score(all_true_classes, all_pred_classes,
                      average = "macro")

        # Save to log
        with open(log_file, "a") as log:
            log.write(f"Epoch {epoch} validation : Balanced Accuracy = {ba:.2f}, Macro F1-score = {f1:.2f} | Loss : {loss:.2f}\n")

        #Prints
        print(f"Validation performance across samples :")
        print(f"Balanced Accuracy Score : {ba:.2f}")
        print(f"Weighted F1 score : {f1:.2f}")

    return epoch_loss, ba, f1


def test(model, test_loader, loss_fn, epoch, figure_path, log_file, device):
    '''Run a trained model on a data loader and return its predictions.

    Used both for the post-hoc validation scoring of saved runs and for the
    final test-set evaluation. Unlike validate(), it returns the raw
    predicted and true classes so the caller can compute its own metrics.

    Arguments
    ---
    model : nn.Module descendant
        Model through which data is passed
    test_loader : DataLoader
        Data with features and labels (must be z-scored like the training data)
    loss_fn : function
        Method of loss calculation
    epoch, figure_path, log_file : unused here, kept for API compatibility
    device : str
        "cuda" or "cpu"

    Returns
    ---
    epoch_loss : float
        Mean loss over batches
    all_pred_classes : list of int
    all_true_classes : list of int
    '''
    model.eval()

    test_loss = 0
    counter = 0

    all_true_classes = []
    all_pred_classes = []
    #samples = []
    with torch.no_grad():
        for batch in test_loader:
            counter += 1

            #Send the input to the device
            X, y = batch['intensity'].to(device), batch['label'].to(device)

            #Compute prediction and loss
            pred = model(X)

            # Loss calculation
            loss = torch.sum(loss_fn(pred, y))
            test_loss += loss.item()

            # Batch metrics - first detach vectors
            pred = pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            class_preds = np.argmax(pred, axis = 1)

            # Save class information
            all_true_classes.extend(y)
            all_pred_classes.extend(class_preds)
            #samples.append(X)

        #Loss and protein metrics
        epoch_loss = test_loss / counter

        ba = balanced_accuracy_score(all_true_classes, all_pred_classes)
        f1 = f1_score(all_true_classes, all_pred_classes,
                      average = "macro")

        #Prints
        print(f"\n Performance across samples : \n")
        print(f"Balanced Accuracy Score : {ba}")
        print(f"Weighted F1 score : {f1}")

    return epoch_loss, all_pred_classes, all_true_classes


def fit(dataset: CIMADataset,               # CIMA Args
        anchor: str,
        K: int,
        ncell: int,
        genes: list,
        save_loc: str,
        nruns: int = 5,
        dendrogram_cutoff: float = 0.4,
        n_val_folds: int = 3,
        maxpool_percentages : list = [1, 5, 10, 20, 50, 100],
        nfilter_selection: list = [3, 4, 5, 6, 7, 8, 9, 10],
        dropout: bool = True,
        dropout_p: float = 0.5,
        l2: float = 1e-4,
        background: bool = False,
        batch_size = 256,
        lr: float = 0.01,
        epochs: int = 20,
        num_workers: int = 8,
        early_stopping = True,
        patience: int = 3,
        seed: int = 12345,
        classification: bool = True):
    '''Fit an S3CIMA Model to a dataset.

    Arguments
    ---
    dataset            : CIMADataset of neighbourhoods
    anchor, K, ncell   : used for logging / naming the results folder
    genes              : marker names, in intensity-column order
    save_loc           : directory in which a timestamped results folder is made
    nruns              : number of independent runs (hyper-parameter draws)
    dendrogram_cutoff  : cosine-distance cut for consensus filter clustering
    n_val_folds        : number of patient-level train/validation folds
    maxpool_percentages: pooling sizes (as % of ncell) cycled through by run.
                         k = max(1, int(p/100 * ncell)), so very small values
                         all collapse to k = 1 ("is there at least one such
                         cell?"), 100 means average over all cells ("how much
                         of the neighbourhood looks like this?").
    nfilter_selection  : candidate numbers of filters, drawn at random per run
    dropout, dropout_p : dropout applied to the pooled filter values
    l2                 : weight decay of Adam
    background         : user's random_ctrl setting (see [FIX-4]: the
                         EFFECTIVE value is read from dataset.random_ctrl)
    batch_size, lr, epochs, num_workers : usual training settings
    early_stopping     : stop a fold when validation loss stalls for `patience`
    seed               : global seed (also decides which patients are test)
    classification     : only classification is implemented

    Returns
    ---
    final_results      : list of consensus-filter metadata dicts selected for
                         plotting (up to 4)
    loss_vectors, val_loss_vectors, accuracy_vectors, val_accuracy_vectors :
                         dicts "run{r}_fold{f}" -> per-epoch values
    test_subset        : normalised test NormalizedCIMASubset
    folds              : list of (train, val) NormalizedCIMASubset
    result_folder      : path of the results folder
    '''
    set_seed(seed)
    start = time.time()

    # Set nmarkers from len of genes
    nmarkers = len(genes)

    # Set device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computation device: {device}\n")

    # Results folder
    now = datetime.now()
    today = datetime.today()
    current_time = now.strftime("%H_%M_%S")
    current_day = today.strftime("%d_%m_%Y")

    result_folder = f"{save_loc}/results_Anchor{anchor}_K{K}_{current_day}_{current_time}"
    os.makedirs(result_folder, exist_ok=True)

    # Generate the datasets and set classes
    test_subset, folds = make_patient_stratified_splits(dataset,
                                                        n_val_folds=n_val_folds,
                                                        seed=seed)
    # With a background control, the extra class is appended last.
    if dataset.random_ctrl:
        n_classes = len(np.unique(dataset.label)) + 1
    else:
        n_classes = len(np.unique(dataset.label))

    # Initialise loss
    if classification:
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("Not yet implemented regression.")

    # Log model training
    log_file = f"{result_folder}/training_log_{current_day}_{current_time}.txt"
    with open(log_file, "w") as log:
        log.write(f""""[TRAINING LOG] - S3CIMA MODEL
Reporting training results for {anchor} anchor, K = {K}, ncell = {ncell}.
------------------------------------------------------------
\n
Classification - {classification} (regression if False)
Background samples - {dataset.random_ctrl} (if True, expect one more 'background' class)
Number of classes - {n_classes}
\n
------------------------------------------------------------
                  """)

    # ----------------------------------------------------------------
    # 1. Run the S3-CIMA training
    # ----------------------------------------------------------------
    max_ba = 0
    max_f1 = 0

    # Model save log
    save_model = f"{result_folder}/models"
    os.makedirs(save_model, exist_ok=True)\

    # Save loss vectors as well
    loss_vectors, val_loss_vectors = {}, {}
    accuracy_vectors, val_accuracy_vectors = {}, {}
    f1_vectors, val_f1_vectors = {}, {}

    for run in range(nruns):

        # Reset to calculate best model.
        # max_ba is reset once per RUN (not per fold) on purpose: one
        # checkpoint is kept per run, namely the fold/epoch with the highest
        # validation balanced accuracy across all of the run's folds.
        max_ba = 0

        # First select the parameters for this run
        # (number of filters = number of learned cell signatures; every filter
        #  always sees ALL markers)
        nfilters = np.random.choice(nfilter_selection)
        print('Number of filters: %d' % nfilters)
        mp = maxpool_percentages[run % len(maxpool_percentages)]
        k = max(1, int(mp / 100. * ncell))
        print('Cells pooled: %d' % k)

        # Set mode + optimiser
        if classification:
            regression = False
        else:
            regression = True
        selection_type = "mean"

        # LOG
        with open(log_file, "a") as log:
            log.write(f"""
Currently on run {run} :

Number of filters - {nfilters}
Number of cells pooled (max pooling) - {k}

Batch Size - {batch_size}
Learning Rate - {lr}
Optimiser - Adam
""")

        # Iterate over the train/validation folds
        count = 0  # fold index
        for (train_data, val_data) in folds:

            # Instantiate model: a FRESH model (new random weights) per fold
            model = CellCNN(nmark=nmarkers,
                        nfilter=nfilters,
                        k=k,
                        n_classes=n_classes,
                        dropout = dropout ,
                        dropout_p = dropout_p ,
                        regression = regression,
                        selection_type = selection_type
                        )
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(),
                            lr=lr,amsgrad = False,
                            weight_decay=l2)

            # Early stopping object
            # NOTE: the boolean argument `early_stopping` is overwritten by the
            # EarlyStopping object here. That still works for later folds
            # because the object is truthy, and a fresh object is created for
            # every fold.
            if early_stopping:
                print("[INFO]: Initializing early stopping")
                early_stopping = EarlyStopping(patience=patience, min_delta=0)

            with open(log_file, "a") as log:
                log.write(f"""
Fold number {count} - {len(train_data)} train samples, {len(val_data)} validation samples
--------------------------------------------------------------------
""")
            # Train model for set number of epochs
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            # Objects to save loss vectors
            loss_vector, val_loss_vector = [], []
            accuracy_vector, val_accuracy_vector = [], []
            f1_vector, val_f1_vector = [], []

            for epoch in range(epochs):

                # Train and validate functions
                train_loss, train_ba, train_f1 = train(model, train_loader, loss_fn, optimizer, epoch, log_file, device)
                val_loss, val_ba, val_f1 = validate(model, val_loader, loss_fn, epoch, log_file, device)

                loss_vector.append(train_loss)
                val_loss_vector.append(val_loss)
                accuracy_vector.append(train_ba)
                val_accuracy_vector.append(val_ba)
                f1_vector.append(train_f1)
                val_f1_vector.append(val_f1)

                # NOTE: the break happens before the checkpoint check below,
                # so the epoch that triggers early stopping is never saved.
                if early_stopping:
                    early_stopping(val_loss)
                    if early_stopping.early_stop:
                        break

                # ------------------------------------------------------
                # [FIX-1] Checkpoint selection compared val vs. TRAIN accuracy
                # ------------------------------------------------------
                # Problem: the condition compares the validation balanced
                #   accuracy with max_ba, but max_ba was then set to the
                #   TRAINING balanced accuracy. The saved "best model" was
                #   therefore whichever fold/epoch last beat a training score,
                #   not the one with the best validation score. The checkpoint
                #   also did not record which fold it came from, so it could
                #   later be re-scored on patients it was trained on (see
                #   [FIX-2]).
                #
                # Fix: store the validation accuracy in max_ba, and record the
                #   fold (and epoch) in the checkpoint.
                #
                # Original code (kept for reference):
                # if val_ba > max_ba:
                #     max_ba = train_ba
                if val_ba > max_ba:
                    max_ba = val_ba

                    # From this best model
                    checkpoint = {
                        'run': run,
                        'fold': count,     # [FIX-1] validation fold of this checkpoint
                        'epoch': epoch,    # [FIX-1] epoch of this checkpoint
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ba': val_ba,
                        'f1' : val_f1,
                        "nmarkers": nmarkers,
                        "nfilter": nfilters,
                        "maxpool_percentage": mp,
                        "k": k,
                        "n_classes": n_classes,
                        "dropout": dropout ,
                        "dropout_p": dropout_p ,
                        "regression": regression,
                        "selection_type": selection_type,
                    }
                    torch.save(checkpoint, f"{save_model}/best_model_run{run}.pth")
                # ------------------------------------------------------

                # Keep per-epoch curves, keyed by run and fold
                loss_vectors[f"run{run}_fold{count}"] = loss_vector
                val_loss_vectors[f"run{run}_fold{count}"] = val_loss_vector
                accuracy_vectors[f"run{run}_fold{count}"] = accuracy_vector
                val_accuracy_vectors[f"run{run}_fold{count}"] = val_accuracy_vector
                f1_vectors[f"run{run}_fold{count}"] = f1_vector
                val_f1_vectors[f"run{run}_fold{count}"] = val_f1_vector

            # Fold number
            count += 1

    end = time.time()

    with open(log_file, "a") as log:
                log.write(f"""
--------------------------------------------------------------------
Training finished !
Time = {(end-start)/60:.3f} minutes
--------------------------------------------------------------------

Now re-running best models on the validation fold each was selected on to
identify the most discriminative filters across runs:
""")

    print(f"Training time: {(end-start)/60:.3f} minutes\n")
    # print(len(predictions_across_embds))
    print("Model complete")

    # ----------------------------------------------------------------
    # 2. Get discriminative filters
    # ----------------------------------------------------------------

    # [FIX-2] Previously one validation fold was picked at random, and its
    # z-scoring was stripped by taking `.subset`. All normalised folds are now
    # passed so each run is scored on its own fold.
    #
    # Original code (kept for reference):
    # # Now identify discriminative filters on the validation set
    # # For now, pick one of the validation sets randomly
    # validation_dataset = random.choice([v[1]for v in folds])
    # validation_dataset = validation_dataset.subset

    # Chance-level threshold: (C-1)/C, e.g. 0.5 for 2 classes.
    accur_thres = (n_classes - 1) / n_classes
    filter_results = get_discriminative_filters(
        save_model = save_model,
        nruns = nruns,
        # validation_subset = validation_dataset,   # [FIX-2] original argument
        folds = folds,
        loss_fn = loss_fn,
        n_classes = n_classes,
        nmarkers = nmarkers,
        lr = lr,
        batch_size = batch_size,
        num_workers = num_workers,
        log_file = log_file,
        device = device,
        accur_thres = accur_thres,
        dendrogram_cutoff = dendrogram_cutoff,
        result_folder = result_folder,
        genes=genes,
        # ----------------------------------------------------------
        # [FIX-4] Background flag must match the dataset
        # ----------------------------------------------------------
        # Problem: CIMADataset silently sets random_ctrl = False when
        #   anchor == "BG", but the user's original random_ctrl value (default
        #   True in run_s3cima) was passed here. n_classes was then 2 (no
        #   background class) while get_discriminative_filters believed class
        #   n_classes-1 = 1 was "background" and discarded every filter voting
        #   for the second REAL condition. Only filters favouring condition 0
        #   could ever be reported.
        #
        # Fix: use the effective value stored on the dataset.
        #
        # Original line (kept for reference):
        # has_background=background,
        has_background=dataset.random_ctrl,
        # ----------------------------------------------------------
        classification=classification,
    )

    with open(log_file, "a") as log:
                log.write(f"""
--------------------------------------------------------------------
Discriminative filters found
A total of {len(filter_results['meta'])} filter clusters identified.
--------------------------------------------------------------------

Running the models producing the top discriminative filters on test to
report classification:
""")

    # ----------------------------------------------------------------
    # 3. Rank discriminative filters on the held-out test patients
    # ----------------------------------------------------------------
    # How to read this step
    # ---------------------
    # The deliverable of S3-CIMA is the filters themselves: their marker
    # weights and the cells/neighbourhoods they pick out. The classification
    # accuracy is only the means by which the filters were learned, not the
    # end product.
    #
    # Ranking by test accuracy is therefore a GENERALISATION filter: among
    # the filters that already passed the validation-based steps above, we
    # prefer those whose source model also separates the conditions on
    # patients it never saw during training or checkpoint selection. These are
    # the signatures most likely to reflect a reproducible biological effect
    # rather than idiosyncrasies of the training patients.
    #
    # Consequences for interpretation:
    #   - Because the test patients are used to CHOOSE which filters are
    #     reported, the "Test balanced accuracy" printed below is a selection
    #     criterion, not an unbiased performance estimate. Do not report it as
    #     the model's expected accuracy on new patients.
    #   - Test accuracy is a property of the whole SOURCE MODEL (run), not of
    #     the individual filter. Filters from the same run always share the
    #     same test accuracy and therefore tend to be selected together.
    #   - The test set is only one patient per condition, so the ranking
    #     depends on which patients the seed held out. Re-running with several
    #     seeds and looking for filters / cell types that recur is the best
    #     check that a signature generalises.
    filtered_meta = [m for m in filter_results["meta"] if m["discriminative"]]

    t_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    final_results = []
    counter = 0
    for r in filtered_meta:

        run_of_origin = r["source_run"]

        ckpt_path = f"{save_model}/best_model_run{run_of_origin}.pth"
        if not os.path.exists(ckpt_path):
            print(f"[WARN] No checkpoint for run {run_of_origin}, skipping.")
            continue

        params = torch.load(ckpt_path, map_location=device, weights_only=False)

        model = CellCNN(
            nmark = params["nmarkers"],
            nfilter = params["nfilter"],
            k = params["k"],
            n_classes = params["n_classes"],
            dropout = params["dropout"],
            dropout_p = params["dropout_p"],
            regression = params["regression"],
            selection_type = params["selection_type"],
        )
        model.load_state_dict(params["model_state_dict"])
        model.to(device)

        # Whole source model scored on the test patients (z-scored with the
        # training statistics, see [FIX-3] in utils/datasets.py)
        test_loss, preds, trues = test(
            model, t_loader, loss_fn, 0, result_folder, log_file, device
        )

        ba = balanced_accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average = "macro")
        print(f"Top model Test balanced accuracy: {ba:.4f}")
        print(f"Top model Test F1 Score: {f1:.4f}")

        # Keep filters whose source model beats chance on unseen patients
        if ba > accur_thres:
             r['test_ba'] = ba
             final_results.append(r)
             counter += 1

        with open(log_file, "a") as log:
            log.write(f"\n[Top model run {run_of_origin}, {r['filter_diff']:.2f} discriminative score] - "
                      f"Test balanced accuracy: {ba:.2f}, Test Macro F1 Score: {f1:.2f}\n")
            log.write(f'Found {counter} discriminative filters with test balanced accuracy above {accur_thres:.2f}\n')

    if counter == 0:
        # No filter generalised above chance: fall back to every filter that
        # passed the validation-based filter_diff check.
        print(f"[INFO] No discriminative filters had test balanced accuracy above {accur_thres:.2f}.")
        print(f"Returning all discriminative filters")

        with open(log_file, "a") as log:
            log.write(f"[INFO] No discriminative filters had test balanced accuracy above {accur_thres:.2f}.")

        final_results = filtered_meta

    # Else return the top n or 4 filters with highest test ba
    else:
        if counter > 4:
            final_results = sorted(final_results, key=lambda x: x["test_ba"], reverse=True)[:4]
            with open(log_file, "a") as log:
                log.write(f"[INFO] Top 4 discriminative filters saved.")
        else:
            with open(log_file, "a") as log:
                log.write(f"[INFO] Top {len(final_results)} discriminative filters saved.")

    return final_results, loss_vectors, val_loss_vectors, accuracy_vectors, val_accuracy_vectors, test_subset, folds, result_folder
