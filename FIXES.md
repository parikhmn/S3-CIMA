# Fixes to S3-CIMA (v0.1.15)

This file summarizes the corrections made to this fork. In the source, each
change is marked with a `[FIX-n]` comment block. The original line(s) are kept
**commented out** directly above the replacement, so you can audit the change
or revert it.

| ID | File | Problem | Effect on outputs |
|----|------|---------|-------------------|
| FIX-1 | `src/s3cima/utils/model.py` (`fit`) | Checkpoint check compared validation accuracy against a stored value, but stored the **training** accuracy (`max_ba = train_ba`). The checkpoint did not record its fold. | The "best model" per run was not the best on validation. Now stores `val_ba`, and the checkpoint records `fold` and `epoch`. |
| FIX-2 | `src/s3cima/utils/model.py` (`fit`, `get_discriminative_filters`) | Post-hoc run scoring and `filter_diff` used `validation_dataset.subset`, which drops z-scoring, so models were scored on **raw** intensities. They also used one randomly chosen fold, whose patients could be training data for checkpoints from other folds. | Run filtering (`accur_thres`), the `val_ba` in the metadata and `filter_diff` are now computed on z-scored data from **each run's own validation fold**. New metadata field: `source_fold`. |
| FIX-3 | `src/s3cima/utils/datasets.py` (`make_patient_stratified_splits`) | The test set was z-scored with statistics from the two test patients only. | The test set now uses the same statistics as training (all non-test patients). |
| FIX-4 | `src/s3cima/utils/model.py` (`fit`) | With `anchor="BG"`, the dataset silently disables `random_ctrl`, but the user's value (default `True`) was passed on as `has_background`. Every filter voting for the second **real** condition was discarded as "background". | Uses `dataset.random_ctrl`. Only affects BG runs with `random_ctrl=True`. |
| FIX-5 | `src/s3cima/utils/preprocess.py` (`process_csv`) | Docstring said `filter_cell_types` **keeps** the listed types. The code **removes** them. | Documentation only. Behavior is unchanged (types are removed). |
| FIX-6 | `src/s3cima/utils/model.py` (`get_discriminative_filters`) | Message said "Keeping top 3" when no run passed the threshold, but all runs are kept. | Console message only. |

## Documented but intentionally not changed

- **Filters reported are ranked by test accuracy.** This is kept on purpose.
  The deliverable is the filter components (marker weights, selected cells),
  not a performance number. Ranking by test accuracy prefers signatures whose
  source model generalizes to unseen patients. As a result, the reported
  test balanced accuracy is a selection criterion, not an unbiased estimate.
  It is also a property of the whole source model, not of an individual
  filter. See the comment block at step 3 of `fit()`.
- **"Selected cells" scoring** (`utils/plot.py`) omits the filter bias and
  averages over all cells, whatever top-k pooling the model used. Selection
  is per neighborhood, not per cell.
- **Enrichment report** compares selected and non-selected cells within each
  sample. It is not the paper's enrichment score (ES).
- Normalization statistics for the training folds include the validation
  patients. This leaks only the per-marker scale, not labels, and is the same
  for every fold.
- The epoch that triggers early stopping is never checkpointed.
- Regression ("functional enrichment") is not implemented.

## Not backward compatible

`get_discriminative_filters()` now takes `folds=` (the list returned by
`make_patient_stratified_splits`) instead of `validation_subset=`. Checkpoints
from older runs have no `fold` key. They are scored on fold 0, with a warning.
