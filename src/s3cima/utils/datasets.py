"""Utils specific to the S3-CIMA Dataset classes.

Overview
--------
This module turns a per-cell table (one row per segmented cell, with marker
intensities and x/y coordinates) into the "multi-cell inputs" that the
S3-CIMA / CellCNN model is trained on.

    per-cell table  ──►  CIMADataset  ──►  one sample = one neighbourhood
                                             (ncell cells × M markers)

A neighbourhood is built around an *anchor* cell: we look up the anchor's K
nearest neighbours in the same image and randomly keep `ncell` of them. The
anchor cell itself is NOT included. The neighbourhood inherits the
condition label of the patient/sample it comes from.

Space therefore enters the model ONLY through which cells are grouped
together. The model never sees coordinates, distances or the arrangement of
cells within the neighbourhood; to the model a neighbourhood is an
unordered bag of cells.

The module also provides:
  - NormalizedCIMASubset : z-scores each neighbourhood on the fly
  - spatial_collate_fn   : batches neighbourhoods for a DataLoader
  - get_norm_stats       : per-marker mean/std used for z-scoring
  - make_patient_stratified_splits : patient-level train/val/test splits

Notes on fixes
--------------
Changes made to the original code are marked with ``[FIX-n]`` blocks. The
original line(s) are kept commented out directly above the replacement so the
change is easy to audit. See FIXES.md at the repository root for a summary.
"""

# Imports
import os
import argparse
import pickle
import random
import time
import json
from typing import Union
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from scipy.spatial import cKDTree


# Functions ---------------------------------------------------------------

# TODO add a calculation of an outlier score - this will help once we
# implement the outlier sampling.

class CIMADataset(Dataset):
    """
    One sample is one anchor cell + its K nearest neighbours' intensities.

    How samples are built
    ---------------------
    For every image separately:
      1. A KD-tree is built on the x/y coordinates of the image's cells.
      2. Anchor cells are chosen:
           - anchor == <cell type> : every cell of that type is an anchor
                                     ("local enrichment" in the paper).
           - anchor == "BG"        : up to `bg_sets` random cells per image are
                                     anchors ("global enrichment").
      3. For each anchor, the K nearest neighbours are retrieved (excluding
         the anchor itself) and `ncell` of them are randomly subsampled.
         The resulting (ncell, M) intensity matrix is one training sample.
      4. If `random_ctrl` is True (and anchor != "BG"), an equal number of
         neighbourhoods centred on random NON-anchor cells are added with an
         extra "background" class label. This lets the model (and you) tell
         whether a signature is specific to the anchor's niche or just a
         general difference between conditions.

    Each sample is a dict with keys:
        intensity : FloatTensor (ncell, M) — raw (un-normalised) intensities
        cellids   : list of the ncell cell IDs in the neighbourhood
        pat       : patient/sample ID the neighbourhood comes from
        label     : encoded condition label (or the background class)
        task      : "classification" or "regression"
        is_ctrl   : True for background-control neighbourhoods

    Parameters
    ----------
    anchor      : str   — anchor cell type label, or "BG" for random anchors
    image       : (N,) array of image IDs
    pat         : (N,) array of patient IDs
    intensity   : (N, M) float array of marker intensities
    ct          : (N,) array of cell type strings
    x, y        : (N,) float arrays of spatial coordinates
    cellid      : (N,) array of cell IDs
    K           : int  — number of nearest neighbours
    ncell       : int - number of K neighbours to subsample (CellCNN input)
    label       : (N,) array of labels (already integer-encoded for classification)
    random_ctrl : bool — if True, expose random-anchor controls as well
    task        : str  — "classification" or "regression", inferred from label type
    seed        : int  — random seed
    bg_sets     : int  — number of random anchors per image when anchor == "BG"
    """

    def __init__(
        self,
        anchor: str,
        image, pat, intensity, ct, x, y, cellid, label,
        K: int,
        ncell : int,
        random_ctrl: bool = False,
        task: str = "classification",
        seed: int = 420,
        bg_sets: int = 500,
    ):
        # Task input check
        assert task in ("classification", "regression"), \
            "task must be 'classification' or 'regression'"

        super().__init__()
        self.K = K
        self.ncell = ncell
        self.label = label
        self.bg_sets = bg_sets

        self.random_ctrl = random_ctrl
        # Dedicated random generator so neighbourhood sampling is reproducible
        # for a given seed, independently of numpy's global RNG.
        self.rng = np.random.default_rng(seed)

        # In "BG" (global) mode the anchors are already random cells, so a
        # separate random-anchor background class would be meaningless.
        # NOTE: this silently overrides the user's random_ctrl. Downstream code
        # must therefore read `dataset.random_ctrl` (the effective value) rather
        # than the argument passed in — see [FIX-4] in utils/model.py.
        if anchor == "BG":
            random_ctrl = False # Overrides for BG
        self.random_ctrl = random_ctrl

        # resolve label type
        self.task = task
        print(f"Selected task : {self.task}")

        # --- collect anchor indices per image, build KD-trees once --------
        self.samples = []

        # Set control label. Real conditions are encoded 0..C-1, so the
        # background class gets the next free integer, C.
        if random_ctrl:
            ctrl_label = len(set(self.label))
            encoded_ctrl = self._encode_label(ctrl_label)


        # Neighbourhoods are always built WITHIN a single image: neighbours are
        # never taken across images.
        for img_id in np.unique(image):
            mask = image == img_id
            ix   = np.where(mask)[0]

            # Slice out this image's cells
            img_x = x[ix]
            img_y = y[ix]
            img_int = intensity[ix]
            img_ct = ct[ix]
            img_cellid = cellid[ix]
            # Assumes one patient per image (takes the first patient ID found)
            img_pat = np.unique(pat[ix])[0]
            img_labels = label[ix]

            # KD-tree over (x, y) for fast nearest-neighbour look-ups
            coords = np.stack([img_x, img_y], axis=1)
            tree   = cKDTree(coords)

            if anchor == "BG":
                # Global mode: choose up to bg_sets random anchor cells.
                # If bg_sets >= number of cells in the image, EVERY cell
                # becomes an anchor and neighbourhoods overlap heavily.
                if len(np.unique(img_labels)) == 1:
                    anchor_idx = self.rng.choice(
                        len(img_ct), size=min(self.bg_sets, len(img_ct)), replace=False
                    )
                else:
                    # Rare case: an image containing cells with several labels.
                    # Split the anchor budget evenly between labels.
                    unique_labels = np.unique(img_labels)
                    quota = max(1, self.bg_sets // len(unique_labels))
                    parts = []
                    for lbl in unique_labels:
                        lbl_idx = np.where(img_labels == lbl)[0]
                        parts.append(
                            self.rng.choice(lbl_idx, size=min(quota, len(lbl_idx)), replace=False)
                        )
                    anchor_idx = np.concatenate(parts)
            else:
                # Local mode: every cell of the requested type is an anchor
                anchor_idx = np.where(img_ct == anchor)[0]

            # Selecting the actual anchor cells
            for idx in anchor_idx:
                knn_int, knn_ids = self._query(idx, tree, img_int, img_cellid)
                self.samples.append({
                    "intensity": torch.tensor(knn_int, dtype=torch.float32),
                    "cellids":   knn_ids,
                    "pat":       img_pat,
                    "label":     self._encode_label(img_labels[idx]),
                    "task":      self.task,
                    "is_ctrl":   False,
                })
            # Background control cells (aka non-anchor cells)
            if random_ctrl:
                # One random non-anchor centre per anchor, so the background
                # class is the same size as the anchor classes in each image.
                # NOTE: rng.choice(..., replace=False) raises if an image has
                # more anchors than non-anchor cells.
                all_idx    = np.arange(len(img_ct))
                non_anchor = np.delete(all_idx, anchor_idx)
                ctrl_idx   = self.rng.choice(
                    non_anchor, size=len(anchor_idx), replace=False
                )

                for idx in ctrl_idx:
                    knn_int, knn_ids = self._query(idx, tree, img_int, img_cellid)
                    self.samples.append({
                        "intensity": torch.tensor(knn_int, dtype=torch.float32),
                        "cellids":   knn_ids,
                        "pat":       img_pat,
                        "label":     encoded_ctrl,
                        "task":      self.task,
                        "is_ctrl":   True,
                    })



    # label query and encoding
    def _encode_label(self, val) -> torch.Tensor:
        """Convert a label to a tensor: long for classification, float for regression."""
        if self.task == "classification":
            # expects val to already be an integer class index,
            # or a string pre-mapped to an int by the user upstream
            return torch.tensor(int(val), dtype=torch.long)
        else:
            return torch.tensor(float(val), dtype=torch.float32)


    def _query(self, idx, tree, img_int, img_cellid):
        """Return the intensities and IDs of `ncell` random cells drawn from
        the K nearest neighbours of cell `idx`.

        K+1 neighbours are requested because the closest "neighbour" of a cell
        is the cell itself (distance 0); that first hit is dropped so the
        anchor is never part of its own neighbourhood.
        """
        _, nn_ix = tree.query(tree.data[idx], k=self.K + 1)
        nn_ix = nn_ix[1:]  # drop the anchor itself

        assert self.ncell <= self.K, (
        f"ncell ({self.ncell}) must be <= K ({self.K}): "
        f"cannot sample more cells than the number of nearest neighbours retrieved."
    )
        # Random subsample of ncell out of the K neighbours. This acts as data
        # augmentation-like noise and is why selected-cell counts in the
        # outputs come in multiples of ncell.
        nn_ix = self.rng.choice(nn_ix, size=self.ncell, replace=False)

        return img_int[nn_ix], img_cellid[nn_ix].tolist()


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        # Returns the RAW (un-normalised) sample. Normalisation is applied by
        # NormalizedCIMASubset, so always go through that wrapper when feeding
        # a trained model.
        return self.samples[idx]


class NormalizedCIMASubset(Dataset):
    def __init__(self, subset: Subset, mean: torch.Tensor, std: torch.Tensor):
        """
        Wraps a Subset of CIMADataset to apply Z-score normalization.

        Every neighbourhood is transformed marker-wise as (x - mean) / std
        when it is accessed. The model is trained on these z-scored values,
        so any later evaluation of a trained model MUST also use z-scored
        inputs computed with the SAME mean/std (see [FIX-2] and [FIX-3]).

        Interpretation note: after z-scoring a below-average marker value is
        negative, so a filter can respond to cells partly because they LACK a
        marker (negative weight × negative value = positive contribution).

        Attributes
        ----------
        subset : torch Subset of the raw CIMADataset (un-normalised!)
        mean   : (M,) per-marker mean
        std    : (M,) per-marker standard deviation
        """
        self.subset = subset
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Retrieve the dictionary from the underlying CIMADataset
        sample = self.subset[idx]
        # clone() so the raw tensor stored in the dataset is never modified
        x = sample["intensity"].clone()
        x = (x - self.mean) / self.std

        # Return the updated dictionary
        return {
            "intensity": x,
            "label":     sample["label"],
            "pat":       sample["pat"],
            "task":      sample["task"],
            "is_ctrl":   sample["is_ctrl"],
            "cellids":   sample["cellids"]
        }


def spatial_collate_fn(batch):
    """Collate a list of neighbourhood dicts into one batch dict.

    Tensors (intensity, label) are stacked; metadata (cell IDs, patient IDs,
    control flags) are kept as Python lists.
    """
    return {
        "intensity": torch.stack([s["intensity"] for s in batch]),  # (B, K, M)
        "cellids":   [s["cellids"] for s in batch],
        "pat":       [s["pat"]     for s in batch],
        "label":     torch.stack([s["label"]     for s in batch]),  # (B,) long or float32
        "is_ctrl":   [s["is_ctrl"] for s in batch],
    }


def get_norm_stats(dataset: CIMADataset, train_idx: list):
    """
    Calculates mean and std of intensities across all training samples.

    The statistics are computed over every cell of every listed
    neighbourhood. Because neighbourhoods overlap, cells appearing in many
    neighbourhoods are counted many times (i.e. these are neighbourhood-
    weighted statistics, not plain per-cell statistics).

    Returns
    -------
    mean, std : (M,) tensors. Markers with zero variance get std = 1 so the
                division is safe (such markers become constant 0 after
                centring and can never influence a filter).
    """
    # Collect intensities from the specified training indices
    train_intensities = torch.stack([dataset.samples[i]["intensity"] for i in train_idx])

    # Calculate marker-wise stats: flatten (n_neigh, ncell, M) -> (n_neigh*ncell, M)
    flat_intensities = train_intensities.view(-1, train_intensities.shape[-1])

    mean = flat_intensities.mean(dim=0)
    std = flat_intensities.std(dim=0)

    # safety
    std[std == 0] = 1.0

    return mean, std


def make_patient_stratified_splits(
    dataset: CIMADataset,
    n_val_folds: int = 5,
    seed: int = 12345,
):
    """
    Build patient-stratified train/val/test splits with no sample-level leakage,
    stratified by patient-level label.

    Splitting is done by PATIENT, never by neighbourhood: neighbourhoods from
    the same image are strongly correlated (and overlap), so splitting them
    across sets would leak information.

    Strategy
    --------
    - 1 patient per label class is the held-out test set
    - Remaining patients split into n_val_folds folds:
        each fold uses 1 patient per label class as validation, rest as training

    Consequences worth knowing
    --------------------------
    - The test set is ALWAYS exactly one patient per condition, whatever the
      cohort size. The seed decides which ones, so re-running with several
      seeds is the way to check robustness of the results.
    - Patients beyond the first n_val_folds per label are only ever used for
      training (never for validation).

    Normalisation
    -------------
    All non-test patients (train + val of any fold — this is the same set of
    patients for every fold) define the z-scoring statistics. These same
    statistics are now also applied to the test set ([FIX-3]).

    Parameters
    ----------
    dataset     : CIMADataset
    n_val_folds : int - number of train/val splits
    seed        : int

    Returns
    -------
    test_subset : NormalizedCIMASubset
    folds       : list of (train NormalizedCIMASubset, val NormalizedCIMASubset) tuples
    """
    rng = np.random.default_rng(seed)

    # derive patient level + labels
    # exclude control samples (is_ctrl=True) to get the real label per patient
    pat_to_label = {}
    for s in dataset.samples:
        if not s["is_ctrl"]:
            p = s["pat"]
            if p not in pat_to_label:
                pat_to_label[p] = s["label"].item()   # scalar from tensor

    # group patients by their label
    label_to_pats = defaultdict(list)
    for p, lbl in pat_to_label.items():
        label_to_pats[lbl].append(p)

    # Shuffle patients within each label (seeded), then check there are
    # enough patients for 1 test + n_val_folds validation patients.
    for lbl in label_to_pats:
        rng.shuffle(label_to_pats[lbl])
        assert len(label_to_pats[lbl]) >= n_val_folds + 1, (
            f"Label '{lbl}' has only {len(label_to_pats[lbl])} patients — "
            f"need at least n_val_folds+1={n_val_folds + 1}"
        )

    # test hold-out: the first (shuffled) patient of each label
    test_pats = set()
    remaining_pats = {}                        # label --> list of remaining patients
    for lbl, pats in label_to_pats.items():
        test_pats.add(pats[0])
        remaining_pats[lbl] = pats[1:]

    # Patient ID of every neighbourhood (including background controls, which
    # inherit the patient of the image they were drawn from)
    sample_pats = np.array([s["pat"] for s in dataset.samples])
    test_idx = [i for i, p in enumerate(sample_pats) if p in test_pats]

    # ----------------------------------------------------------------------
    # [FIX-3] Test set z-scored with its OWN statistics
    # ----------------------------------------------------------------------
    # Problem: the test set was normalised with a mean/std computed from the
    #   two test patients only, while the models were trained on data
    #   normalised with statistics from all non-test patients. The same raw
    #   intensity therefore mapped to a different z-score at test time than
    #   during training. This (a) makes the test inputs inconsistent with what
    #   the filters learned, and (b) uses information from the test patients
    #   themselves to transform them. A shift in test accuracy could come from
    #   this scaling difference alone.
    #
    # Fix: compute the statistics from all non-test neighbourhoods (exactly
    #   the set used for every training fold below, since train ∪ val of any
    #   fold = all non-test patients) and apply them to the test set.
    #
    # Original code (kept for reference):
    # m_test, s_test = get_norm_stats(dataset, test_idx)
    # test_subset = NormalizedCIMASubset(Subset(dataset, test_idx), m_test, s_test)
    non_test_idx = [i for i, p in enumerate(sample_pats) if p not in test_pats]
    m_test, s_test = get_norm_stats(dataset, non_test_idx)
    test_subset = NormalizedCIMASubset(Subset(dataset, test_idx), m_test, s_test)
    # ----------------------------------------------------------------------
    print(f"Test : {len(test_idx)} samples ({len(test_pats)} patients, "
          f"1 per label: {list(label_to_pats.keys())})")

    # build the splits
    folds = []

    # Check input number of folds is correct
    min_pats_per_label = min(len(pats) for pats in remaining_pats.values())
    min_label = min(remaining_pats, key=lambda l: len(remaining_pats[l]))
    assert n_val_folds <= min_pats_per_label, (
        f"n_val_folds={n_val_folds} exceeds the number of remaining patients for label "
        f"'{min_label}' ({min_pats_per_label} patients after test holdout). "
        f"Reduce n_val_folds to at most {min_pats_per_label}."
    )

    # TODO - improve the fold formation.
    for fold in range(n_val_folds):
        # Fold f validates on the f-th remaining patient of each label and
        # trains on all other remaining patients.
        val_pats = set()
        train_pats = set()
        for lbl, pats in remaining_pats.items():
            val_pat = pats[fold % len(pats)]
            val_pats.add(val_pat)
            train_pats.update(p for p in pats if p != val_pat)

        val_idx = [i for i, p in enumerate(sample_pats) if p in val_pats]
        train_idx = [i for i, p in enumerate(sample_pats) if p in train_pats]

        # NOTE: statistics include the validation patients as well as the
        # training patients. This is a mild form of leakage (only the per-
        # marker scale, not the labels) and is identical for every fold.
        m, s = get_norm_stats(dataset, train_idx + val_idx)

        train_subset = NormalizedCIMASubset(Subset(dataset, train_idx), m, s)
        val_subset = NormalizedCIMASubset(Subset(dataset, val_idx), m, s)
        folds.append((train_subset, val_subset))
        print(
            f"Fold {fold}: "
            f"train={len(train_idx)} samples ({len(train_pats)} patients) | "
            f"val={len(val_idx)} samples ({len(val_pats)} patients)"
        )

    return test_subset, folds
