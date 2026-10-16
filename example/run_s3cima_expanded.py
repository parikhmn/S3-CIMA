"""run_s3cima, expanded in place.

This script performs the SAME analysis as the notebook's CD8 call

    run_s3cima("CD8", image_id, sample_id, intensity, genes, cell_type, x, y,
               cell_id, labels, label_map, K=50, ncell=30, random_ctrl=False,
               n_val_folds=3, batch_size=128, lr=0.001, nruns=10, seed=43, ...)

but with (almost) every function of the package written out inline, top to
bottom, in the order it runs. It is meant to be READ, to understand the
workflow. Use the package itself for real analyses.

It reproduces the package's results exactly (same seed -> same filters,
same selected cells), because every random draw happens in the same order as
in the package. Two consequences of that:
  - a few seemingly redundant lines are kept on purpose (e.g. the validation
    DataLoader uses shuffle=True, and DataLoaders are re-created where the
    package re-creates them: each one draws from torch's random generator);
  - the training text log, progress bars and console prints are left out.

What is still a function or class here, and why
------------------------------------------------
  - process_csv          : input preparation, called BEFORE run_s3cima.
  - CellCNN (as `Net`)   : torch needs an nn.Module; its forward pass and the
                           top-k pooling (select_top_pool) are written inside.
  - Library calls        : torch, numpy, scipy, sklearn, pandas, plotting.

Call counts for the notebook CD8 run (tests/data, 10 runs × 3 folds, up to
20 epochs; measured with a profiler) are given in each section header as
"[package function × N]". Functions called many times are inlined once inside
the loop that repeats them.

Sections
--------
  0. Inputs                         (process_csv — not part of run_s3cima)
  1. Seeds                          (set_seed)
  2. Neighbourhoods                 (CIMADataset)
  3. Patient splits + z-scoring     (make_patient_stratified_splits, get_norm_stats,
                                     NormalizedCIMASubset)
  4. Training                       (fit: CellCNN, EarlyStopping, train, validate)
  5. Consensus filters              (get_discriminative_filters, test,
                                     _compute_filter_diff, _single_filter_output)
  6. Rank filters on test patients  (end of fit)
  7. Filter weight heatmap          (plot_filter_weights)
  8. Selected neighbourhoods        (get_high_response_cells_test / _train)
  9. Selected-cell CSVs and maps    (save_stats)
 10. Enrichment reports             (enrichment_summary)

Usage (from the repository root, with the package installed):
    python example/run_s3cima_expanded.py [output_folder]
"""

import os
import sys
import glob
import json
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from s3cima.utils.preprocess import process_csv


# =============================================================================
# 0. INPUTS — one row per segmented cell        [process_csv × 1, before run]
# =============================================================================
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = sys.argv[1] if len(sys.argv) > 1 else "."

(intensity, genes, x, y, cell_id, cell_type,
 sample_id, image_id, labels, label_map) = process_csv(
    markers_path      = f"{REPO}/tests/data/cima_test_markers.csv",
    meta_path         = f"{REPO}/tests/data/cima_test_meta.csv",
    x_col             = "spatial_dim0",
    y_col             = "spatial_dim1",
    cell_id_col       = "cell_id",
    cell_type_col     = "cell_type",
    sample_id_col     = "sample",
    image_id_col      = "image_id",
    condition_col     = "condition",
    filter_cell_types = ["Schwann_cells", "Muscular_cells"],  # REMOVED from the data
)
# intensity: (N cells, M markers)   labels: integer condition per cell (0..C-1)
# label_map: {0: "early", 1: "normal"}  — maps integers back to names

# run_s3cima arguments (notebook values; everything else = package defaults)
anchor              = "CD8"     # or "BG" for random anchors (global analysis)
K                   = 50        # nearest neighbours per anchor
ncell               = 30        # of those K, how many are randomly kept
random_ctrl         = False     # add a "background" class of random centres?
task                = "classification"
nruns               = 10
dendrogram_cutoff   = 0.4
n_val_folds         = 3
maxpool_percentages = [0.01, 1, 5, 20, 100]
nfilter_selection   = [3, 4, 5, 6, 7, 8, 9, 10]
filter_threshold    = 0.9       # top 10% of neighbourhoods are "selected"
dropout, dropout_p  = True, 0.5
bg_sets             = 500
batch_size          = 128
lr                  = 0.001
l2                  = 1e-4
epochs              = 20
num_workers         = 0
early_stopping      = True
patience            = 3
seed                = 43


# =============================================================================
# 1. SEEDS                                                     [set_seed × 2]
# =============================================================================
# run_s3cima and fit() each call set_seed(seed); both calls are identical.
def _set_seed(s):
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(s)
_set_seed(seed)   # (this tiny helper is the one exception to "inline everything")


# =============================================================================
# 2. NEIGHBOURHOODS                                       [CIMADataset × 1,
#                                     _query × 28, _encode_label × 28 (CD8 run)]
# =============================================================================
# One sample = one anchor cell's neighbourhood: ncell of its K nearest
# neighbours (anchor excluded), as a (ncell, M) intensity matrix, labelled
# with the condition of its patient. Space enters ONLY through which cells
# are grouped; the model never sees coordinates.
rng = np.random.default_rng(seed)      # the dataset's own generator
if anchor == "BG":
    random_ctrl = False                # random anchors already = background
n_conditions = len(set(labels))
ctrl_label = n_conditions              # background class index, if used

samples = []                           # the whole dataset: a list of dicts
for img_id in np.unique(image_id):
    ix = np.where(image_id == img_id)[0]
    img_int, img_ct, img_cellid = intensity[ix], cell_type[ix], cell_id[ix]
    img_pat = np.unique(sample_id[ix])[0]          # one patient per image
    img_labels = labels[ix]
    tree = cKDTree(np.stack([x[ix], y[ix]], axis=1))

    # --- choose anchor cells --------------------------------------------
    if anchor == "BG":
        if len(np.unique(img_labels)) == 1:
            anchor_idx = rng.choice(len(img_ct), size=min(bg_sets, len(img_ct)),
                                    replace=False)
        else:
            quota = max(1, bg_sets // len(np.unique(img_labels)))
            anchor_idx = np.concatenate([
                rng.choice(np.where(img_labels == l)[0],
                           size=min(quota, (img_labels == l).sum()), replace=False)
                for l in np.unique(img_labels)])
    else:
        anchor_idx = np.where(img_ct == anchor)[0]   # every cell of that type

    # --- one neighbourhood per anchor (+ optional background centres) ----
    centres = [(i, False) for i in anchor_idx]
    if random_ctrl:
        non_anchor = np.delete(np.arange(len(img_ct)), anchor_idx)
        ctrl_idx = rng.choice(non_anchor, size=len(anchor_idx), replace=False)
        centres += [(i, True) for i in ctrl_idx]

    for idx, is_ctrl in centres:
        # _query: K+1 nearest (the first hit is the cell itself, dropped),
        # then keep a random ncell of the K neighbours
        _, nn_ix = tree.query(tree.data[idx], k=K + 1)
        nn_ix = rng.choice(nn_ix[1:], size=ncell, replace=False)
        samples.append({
            "intensity": torch.tensor(img_int[nn_ix], dtype=torch.float32),
            "cellids":   img_cellid[nn_ix].tolist(),
            "pat":       img_pat,
            # _encode_label: class index as a long tensor
            "label":     torch.tensor(int(ctrl_label if is_ctrl else img_labels[idx]),
                                      dtype=torch.long),
            "task":      task,
            "is_ctrl":   is_ctrl,
        })
        # NOTE: in the package, all anchors of an image are added first and
        # then all background centres; the order above is the same.

n_classes = n_conditions + (1 if random_ctrl else 0)


# =============================================================================
# 3. PATIENT SPLITS + Z-SCORING          [make_patient_stratified_splits × 1,
#                     get_norm_stats × 4, NormalizedCIMASubset × 7 (CD8 run)]
# =============================================================================
# fit() starts here: set_seed again, then create the results folder.
_set_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
stamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
result_folder = f"{SAVE_PATH}/results_expanded_Anchor{anchor}_K{K}_{stamp}"
model_dir = f"{result_folder}/models"
os.makedirs(model_dir, exist_ok=True)

split_rng = np.random.default_rng(seed)

# Patient -> condition (from non-background samples), grouped by condition
pat_to_label = {}
for s in samples:
    if not s["is_ctrl"] and s["pat"] not in pat_to_label:
        pat_to_label[s["pat"]] = s["label"].item()
label_to_pats = defaultdict(list)
for p, l in pat_to_label.items():
    label_to_pats[l].append(p)
for l in label_to_pats:
    split_rng.shuffle(label_to_pats[l])
    assert len(label_to_pats[l]) >= n_val_folds + 1

# Test set: the first (shuffled) patient of each condition — always exactly
# one patient per condition, whatever the cohort size.
test_pats = {pats[0] for pats in label_to_pats.values()}
remaining = {l: pats[1:] for l, pats in label_to_pats.items()}
sample_pats = np.array([s["pat"] for s in samples])


def norm_stats(idx):
    """get_norm_stats: per-marker mean/std over every cell of every listed
    neighbourhood (overlapping neighbourhoods count cells several times)."""
    flat = torch.stack([samples[i]["intensity"] for i in idx]).view(-1, len(genes))
    m, s = flat.mean(dim=0), flat.std(dim=0)
    s[s == 0] = 1.0
    return m, s


def zscored(idx, m, s):
    """NormalizedCIMASubset: the same samples, intensities z-scored marker-wise.
    (The package does this lazily on access; doing it up front is equivalent.)"""
    return [dict(samples[i], intensity=(samples[i]["intensity"].clone() - m) / s)
            for i in idx]


# Test set z-scored with the statistics of ALL NON-TEST patients ([FIX-3])
test_idx = [i for i, p in enumerate(sample_pats) if p in test_pats]
non_test_idx = [i for i, p in enumerate(sample_pats) if p not in test_pats]
test_set = zscored(test_idx, *norm_stats(non_test_idx))

# Validation folds: fold f validates on the f-th remaining patient of each
# condition and trains on the others. train ∪ val = all non-test patients in
# every fold, so the z-scoring statistics are the same for every fold.
folds = []
for f in range(n_val_folds):
    val_pats = {pats[f % len(pats)] for pats in remaining.values()}
    train_pats = {p for pats in remaining.values() for p in pats} - val_pats
    val_idx = [i for i, p in enumerate(sample_pats) if p in val_pats]
    train_idx = [i for i, p in enumerate(sample_pats) if p in train_pats]
    m, s = norm_stats(train_idx + val_idx)
    folds.append((zscored(train_idx, m, s), zscored(val_idx, m, s)))
    print(f"Fold {f}: {len(train_idx)} train / {len(val_idx)} val neighbourhoods")


# =============================================================================
# 4. TRAINING                         [CellCNN.__init__ × 30 here (72 total),
#        EarlyStopping × 30, train × 559, validate × 559, forward × 1160 total]
# =============================================================================
class Net(nn.Module):
    """CellCNN. A filter = weights over ALL markers + a bias = a learned cell
    signature. Same layer names as the package, so checkpoints are identical."""

    def __init__(self, nmark, nfilter, k, n_classes):
        super().__init__()
        self.k = k
        # kernel_size=1: each filter scores one cell at a time
        self.conv = nn.Conv1d(nmark, nfilter, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="relu")
        self.dropout = nn.Dropout(p=dropout_p) if dropout else None
        self.fc = nn.Linear(nfilter, n_classes)

    def forward(self, X):                     # X: (B, ncell, M)
        h = F.relu(self.conv(X.permute(0, 2, 1)))           # (B, nfilter, ncell)
        # select_top_pool: mean of the k best-responding cells per filter
        h = torch.topk(h, k=self.k, dim=2).values.mean(dim=2)   # (B, nfilter)
        if self.dropout is not None:
            h = self.dropout(h)               # active only in model.train()
        return self.fc(h)                     # (B, n_classes) logits


loss_fn = nn.CrossEntropyLoss()

for run in range(nruns):
    # --- hyper-parameters for this run -----------------------------------
    nfilters = np.random.choice(nfilter_selection)   # HOW MANY signatures
    mp = maxpool_percentages[run % len(maxpool_percentages)]
    k = max(1, int(mp / 100. * ncell))               # pooling size
    max_ba = 0                                       # best val score of this run

    for fold_idx, (train_data, val_data) in enumerate(folds):
        # --- a fresh model per fold --------------------------------------
        model = Net(len(genes), nfilters, k, n_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=False,
                               weight_decay=l2)
        es_best, es_counter = None, 0                # EarlyStopping state
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers)

        for epoch in range(epochs):
            # --- train(): one pass over the training neighbourhoods ------
            model.train()
            for batch in train_loader:
                X, yb = batch["intensity"].to(device), batch["label"].to(device)
                loss = loss_fn(model(X), yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            # --- validate(): loss + balanced accuracy on val patients ----
            model.eval()
            val_loss, n_batches, trues, preds = 0.0, 0, [], []
            with torch.no_grad():
                for batch in val_loader:
                    X, yb = batch["intensity"].to(device), batch["label"].to(device)
                    pred = model(X)
                    val_loss += loss_fn(pred, yb).item()
                    n_batches += 1
                    trues.extend(yb.cpu().numpy())
                    preds.extend(pred.argmax(dim=1).cpu().numpy())
            val_loss /= n_batches
            val_ba = balanced_accuracy_score(trues, preds)

            # --- EarlyStopping.__call__: stop after `patience` epochs
            #     without improvement of the validation loss -----------------
            if early_stopping:
                if es_best is None:
                    es_best = val_loss
                elif es_best - val_loss > 0:
                    es_best, es_counter = val_loss, 0
                elif es_best - val_loss < 0:
                    es_counter += 1
                if es_counter >= patience:
                    break                 # (this epoch is never checkpointed)

            # --- keep the best fold/epoch of this run ([FIX-1]) ----------
            if val_ba > max_ba:
                max_ba = val_ba
                torch.save({
                    "run": run, "fold": fold_idx, "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ba": val_ba, "nfilter": nfilters, "k": k,
                    "maxpool_percentage": mp,
                }, f"{model_dir}/best_model_run{run}.pth")
# Result: ONE checkpoint per run (nruns models in total).


def load_run(run):
    """Rebuild the saved model of a run (the package does this inline twice)."""
    p = torch.load(f"{model_dir}/best_model_run{run}.pth", map_location=device,
                   weights_only=False)
    net = Net(len(genes), p["nfilter"], p["k"], n_classes)
    net.load_state_dict(p["model_state_dict"])
    return net.to(device), p


def predict(net, data):
    """test(): predicted and true classes (a DataLoader is created per call,
    exactly as in the package)."""
    net.eval()
    t, pr = [], []
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers):
            out = net(batch["intensity"].to(device))
            t.extend(batch["label"].numpy())
            pr.extend(out.argmax(dim=1).cpu().numpy())
    return t, pr


# =============================================================================
# 5. CONSENSUS FILTERS                 [get_discriminative_filters × 1,
#        test × 10, _compute_filter_diff × 56, _single_filter_output × 56]
# =============================================================================
# 5a. Re-score each run on the validation fold its checkpoint came from
#     (normalised, [FIX-2]) and keep its filter weights.
accuracies, run_folds, weights = {}, {}, {}
for run in range(nruns):
    if not os.path.exists(f"{model_dir}/best_model_run{run}.pth"):
        continue
    net, p = load_run(run)
    run_folds[run] = p["fold"]
    t, pr = predict(net, folds[p["fold"]][1])
    accuracies[run] = balanced_accuracy_score(t, pr)
    weights[run] = {
        "conv": net.conv.weight.detach().cpu().numpy().squeeze(-1),  # (nfilter, M)
        "out":  net.fc.weight.detach().cpu().numpy().T,              # (nfilter, C)
        "bias": net.conv.bias.detach().cpu().numpy(),                # (nfilter,)
    }

# 5b. Keep runs at or above chance, (C-1)/C (0.5 for two conditions)
accur_thres = (n_classes - 1) / n_classes
sorted_runs = sorted(accuracies, key=accuracies.get, reverse=True)
passing_runs = [r for r in sorted_runs if accuracies[r] >= accur_thres] or sorted_runs

# 5c. Pool every filter of every kept run (minus background-voting filters)
pooled_conv, pooled_out, pooled_bias, pooled_run = [], [], [], []
for run in passing_runs:
    w = weights[run]
    for f in range(w["conv"].shape[0]):
        if random_ctrl and np.argmax(w["out"][f]) == n_classes - 1:
            continue                       # votes for "background": drop
        pooled_conv.append(w["conv"][f]); pooled_out.append(w["out"][f])
        pooled_bias.append(w["bias"][f]); pooled_run.append(run)
pooled_conv, pooled_out = np.array(pooled_conv), np.array(pooled_out)

# 5d. Cluster filters by cosine distance (pattern across markers, not size)
if len(pooled_conv) == 1:
    clusters = np.array([0])
else:
    clusters = fcluster(linkage(pdist(pooled_conv, metric="cosine"), method="average"),
                        t=dendrogram_cutoff, criterion="distance")

# 5e. Consensus filter = medoid of each cluster
consensus_conv, consensus_out, consensus_bias, meta = [], [], [], []
for cid in np.unique(clusters):
    members = np.where(clusters == cid)[0]
    medoid = members[0] if len(members) == 1 else members[
        np.argmin(cdist(pooled_conv[members], pooled_conv[members], "cosine").mean(axis=1))]
    w = pooled_conv[medoid]
    order = np.argsort(np.abs(w))[::-1]
    consensus_conv.append(w); consensus_out.append(pooled_out[medoid])
    consensus_bias.append(pooled_bias[medoid])
    meta.append({
        "cluster": int(cid), "medoid_filter_idx": int(medoid),
        "source_run": pooled_run[medoid],
        "source_fold": int(run_folds[pooled_run[medoid]]),
        "val_ba": round(accuracies[pooled_run[medoid]], 4),
        "n_members": len(members),       # >1 from different runs = recurrent
        "top_genes": [genes[i] for i in order[:10]],
        "top_weights": [round(float(w[i]), 6) for i in order[:10]],
        "all_gene_weights": {genes[i]: round(float(w[i]), 6) for i in order},
    })

# 5f. filter_diff: on the source run's validation fold, mean pooled response
#     on neighbourhoods of the filter's favoured class minus the others.
#     (_compute_filter_diff -> _single_filter_output, pooling fixed at 100%)
for i, m in enumerate(meta):
    val = folds[m["source_fold"]][1]
    y_true = np.array([int(s["label"].item()) for s in val])
    # ReLU per cell, then mean of all cells (np.sort kept only so the float
    # summation order matches the package exactly)
    resp = np.array([
        np.sort(np.maximum(0, s["intensity"].numpy() @ consensus_conv[i]
                           + consensus_bias[i])).mean()
        for s in val], dtype=np.float64)
    fav = int(np.argmax(consensus_out[i]))              # class this filter votes for
    inside = y_true == fav
    fd = (resp[inside].mean() - resp[~inside].mean()
          if inside.any() and (~inside).any() else 0.0)
    m["filter_diff"] = round(float(fd), 6)
    m["discriminative"] = bool(fd >= 0.0)
    m["_filter_diff_raw"] = float(fd)            # unrounded, for the CSV below

# 5g. Save filter files
filter_dir = f"{result_folder}/filters"
os.makedirs(filter_dir, exist_ok=True)
np.save(f"{filter_dir}/consensus_filter_weights.npy", np.array(consensus_conv))
np.save(f"{filter_dir}/consensus_output_weights.npy", np.array(consensus_out))
np.save(f"{filter_dir}/all_pooled_filter_weights.npy", pooled_conv)
np.save(f"{filter_dir}/all_pooled_output_weights.npy", pooled_out)
pd.DataFrame({"cluster": [m["cluster"] for m in meta],
              "filter_diff": [m.pop("_filter_diff_raw") for m in meta],
              "discriminative": [m["discriminative"] for m in meta]}) \
  .sort_values("filter_diff", ascending=False) \
  .to_csv(f"{filter_dir}/filter_discriminability.csv", index=False)
for m in meta:
    pd.DataFrame({"gene": list(m["all_gene_weights"]),
                  "weight": list(m["all_gene_weights"].values())}) \
      .to_csv(f"{filter_dir}/cluster{m['cluster']}_filter_weights.csv", index=False)
with open(f"{filter_dir}/consensus_filter_meta.json", "w") as fh:
    json.dump(meta, fh, indent=2)
print(f"{len(meta)} consensus filters from {len(pooled_conv)} pooled filters")


# =============================================================================
# 6. RANK DISCRIMINATIVE FILTERS ON THE TEST PATIENTS   [test × 32 (CD8 run)]
# =============================================================================
# The source MODEL of each discriminative filter is scored on the two held-out
# patients; filters whose model beats chance are kept, best 4 reported. This
# picks the signatures that generalise to unseen patients — the test accuracy
# is therefore a selection criterion, and a property of the whole run.
final = []
for m in [m for m in meta if m["discriminative"]]:
    net, _ = load_run(m["source_run"])
    t, pr = predict(net, test_set)
    ba = balanced_accuracy_score(t, pr)
    if ba > accur_thres:
        final.append(dict(m, test_ba=ba))
if not final:
    final = [m for m in meta if m["discriminative"]]
elif len(final) > 4:
    final = sorted(final, key=lambda m: m["test_ba"], reverse=True)[:4]
print("Reported filters:", [m["cluster"] for m in final])


# =============================================================================
# 7. FILTER WEIGHT HEATMAP                            [plot_filter_weights × 1]
# =============================================================================
plot_dir = f"{result_folder}/plots"
os.makedirs(plot_dir, exist_ok=True)
hm = pd.DataFrame({f"Cluster {m['cluster']} (Filter Diff: {m['filter_diff']:.2f})":
                   m["all_gene_weights"] for m in final}).T.fillna(0.0)
hm = hm[hm.abs().max(axis=0).sort_values(ascending=False).index]  # strongest first
zmax = np.abs(hm.values).max()
go.Figure(go.Heatmap(z=hm.values, x=hm.columns.tolist(), y=hm.index.tolist(),
                     colorscale="RdBu_r", zmid=0, zmin=-zmax, zmax=zmax,
                     colorbar=dict(title="Weight"))) \
  .update_layout(title="Filter Cluster Gene Weights", yaxis_autorange="reversed",
                 height=max(300, 120 * len(final))) \
  .write_html(f"{plot_dir}/consensus_filter_weights.html")


# =============================================================================
# 8-10. SELECTED NEIGHBOURHOODS, CSVs/MAPS, ENRICHMENT
#   [get_high_response_cells_test × 1 + _train × 1, save_stats × 2,
#    enrichment_summary × 2]   — the same code runs for "test" and "train"
# =============================================================================
# "train" = fold 0's train + val = all non-test patients.
splits = {"test": test_set, "train": folds[0][0] + folds[0][1]}
condition_names = np.asarray([label_map[l] for l in labels])

for split_name, data in splits.items():
    # ---- 8. Score every non-background neighbourhood with each filter -----
    # Simplified score (documented, not the model's): no bias, mean over ALL
    # cells. A neighbourhood is selected if its score is >= the 90th
    # percentile (computed over both conditions together).
    anchors = [s for s in data if not s["is_ctrl"]]
    split_samples = {s["pat"].item() for s in anchors}
    selected = {}                                       # cluster -> cell IDs
    for m in final:
        w = np.array([m["all_gene_weights"][g] for g in genes], dtype=np.float32)
        scores = np.array([np.sort(np.maximum(0.0, s["intensity"].numpy() @ w)).mean()
                           for s in anchors], dtype=np.float32)
        cut = float(np.quantile(scores, filter_threshold))
        ids = []
        for s, sc in zip(anchors, scores):
            if sc >= cut:
                ids += [c for c in s["cellids"] if c not in ids]  # all cells
        selected[m["cluster"]] = ids

    # ---- 9. Per filter: map per sample + all_selected_cells.csv -----------
    split_dir = f"{plot_dir}/{split_name}"
    keep = np.isin(sample_id, list(split_samples))
    for m in final:
        fdir = f"{split_dir}/filter_{m['cluster']}"
        os.makedirs(fdir, exist_ok=True)
        rows = []
        for sid in np.unique(sample_id[keep]):
            sm = keep & (sample_id == sid)
            sel = np.isin(cell_id[sm], selected[m["cluster"]])
            if not sel.any():
                continue                                 # sample skipped
            fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
            ax.scatter(x[sm][~sel], y[sm][~sel], c="#b0b0b0", s=0.25)
            ax.scatter(x[sm][sel], y[sm][sel], c="#8b0000", s=0.25,
                       label=f"Selected (n={sel.sum()})")
            ax.set_title(f"Filter {m['cluster']} | Sample {sid} | "
                         f"{sel.sum()}/{sm.sum()} cells", fontsize=8)
            ax.legend(markerscale=6, fontsize=7); ax.set_aspect("equal")
            fig.savefig(f"{fdir}/spatial_plot_sample_{sid}.png"); plt.close(fig)
            rows.append(pd.DataFrame({
                "x": x[sm][sel], "y": y[sm][sel], "cell_id": cell_id[sm][sel],
                "cell_type": cell_type[sm][sel],
                "condition_id": condition_names[sm][sel], "sample_id": sid}))
        if rows:
            df = pd.concat(rows, ignore_index=True)
            df["cluster_id"] = m["cluster"]
            df["filter_diff"] = round(m["filter_diff"], 4)
            df.to_csv(f"{fdir}/all_selected_cells.csv", index=False)

    # ---- 10. Enrichment: selected vs. non-selected cells of the same sample
    all_types = np.unique(cell_type)
    panels = []
    for csv in sorted(glob.glob(f"{split_dir}/filter_*/all_selected_cells.csv")):
        df = pd.read_csv(csv)
        cid = int(df["cluster_id"].iloc[0])
        fdiff = round(float(df["filter_diff"].iloc[0]), 4)
        top = df["condition_id"].value_counts()
        majority = str(top.index[0]) if top.iloc[0] / len(df) >= 0.9 else "mixed"

        # (a) cell-type proportions among selected cells -> bar chart
        props = df["cell_type"].value_counts(normalize=True)
        fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(props) + 1.5)), dpi=150)
        ax.barh(props.index[::-1], props.values[::-1], color="#8b0000")
        ax.set_xlabel("Proportion of selected cells")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(f"Filter {cid} | majority condition: {majority} | "
                     f"filter_diff: {fdiff:.4f}", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{split_dir}/filter_{cid}/cell_type_proportions.png",
                    bbox_inches="tight")
        plt.close(fig)

        # (b) per-sample log2((p_sel + ps) / (p_rest + ps)), ps = 1/n_sample,
        #     then mean ± SD across samples (NOT the paper's ES)
        recs = []
        for sid, sdf in df.groupby("sample_id"):
            in_sample = sample_id == str(sid)
            n_sel, n_rest = len(sdf), in_sample.sum() - len(sdf)
            if n_rest <= 0:
                continue
            ps = 1.0 / (n_sel + n_rest)
            for ct in all_types:
                c_sel = (sdf["cell_type"] == ct).sum()
                c_rest = max((cell_type[in_sample] == ct).sum() - c_sel, 0)
                recs.append({"cell_type": ct, "log2_enrichment":
                             np.log2((c_sel / n_sel + ps) / (c_rest / n_rest + ps))})
        summ = (pd.DataFrame(recs).groupby("cell_type")["log2_enrichment"]
                .agg(mean="mean", std="std").reindex(all_types, fill_value=0)
                .sort_values("mean"))
        panels.append((cid, fdiff, majority, summ))

    # (c) one interactive report with a panel per filter
    ncols = 2 if len(panels) == 4 else min(3, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[
        f"Filter {c} | {maj} | diff={d:.3f}" for c, d, maj, _ in panels])
    for i, (c, d, maj, summ) in enumerate(panels):
        fig.add_trace(go.Bar(
            x=summ["mean"], y=summ.index, orientation="h", showlegend=False,
            marker_color=["#8b0000" if v >= 0 else "#2166ac" for v in summ["mean"]],
            error_x=dict(type="data", array=summ["std"].fillna(0).tolist())),
            row=i // ncols + 1, col=i % ncols + 1)
    fig.update_layout(height=max(350, 400 * nrows),
                      title_text=f"Consensus Filter Enrichment Summary ({split_name})")
    fig.write_html(f"{split_dir}/enrichment_summary_report_{split_name}.html",
                   include_plotlyjs="cdn")

print(f"Done. Outputs in {result_folder}")
