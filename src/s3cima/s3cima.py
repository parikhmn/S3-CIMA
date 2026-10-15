"""Main function to run S3-CIMA. See README for usage instructions"""

# Imports

import json
from typing import Union

import numpy as np
import pandas as pd


from s3cima.utils.datasets import CIMADataset
from s3cima.utils.model import fit, set_seed
import s3cima.utils.plot as s3cima_plot 

# Function ---------------------------------------------------------------

def run_s3cima(
    anchor: str,
    image_id, sample_id, intensity, genes, cell_type, x, y, cell_id, label, label_map,
    K: int = 50,
    ncell : int = 30,
    random_ctrl: bool = True,
    task: str = "classification",
    nruns: int = 5, 
    dendrogram_cutoff: float = 0.4,
    n_val_folds: int = 3,              
    maxpool_percentages : list = [0.01, 1, 5, 20, 100],
    nfilter_selection: list = [3, 4, 5, 6, 7, 8, 9, 10],
    filter_threshold: float = 0.9,
    dropout: bool = True,
    dropout_p: float = 0.5,
    bg_sets: int = 500,
    batch_size = 256,
    lr: float = 0.1,
    l2: float = 1e-4,
    epochs: int = 20, 
    num_workers: int = 0,
    early_stopping = True,
    patience: int = 3,
    seed: int = 420,
    save_path: str = ".",
    ):  
    """Runs an S3-CIMA analysis end to end.

    Pipeline
    --------
    1. Build neighbourhoods around anchor cells   (CIMADataset)
    2. Split patients, train `nruns` CellCNN runs,
       pool + cluster filters into consensus filters,
       rank discriminative filters               (utils.model.fit)
    3. Heatmap of consensus filter weights        (plot_filter_weights)
    4. Score neighbourhoods with each reported filter and mark the top
       (1 - filter_threshold) as "selected", for the test patients and for
       all non-test patients                      (get_high_response_cells_*)
    5. Save selected cells, spatial plots and enrichment reports
                                                  (save_stats, enrichment_summary)

    Args
    ----
    anchor            : cell type used as neighbourhood centre, or "BG" for
                        random centres (global analysis)
    image_id, sample_id, intensity, genes, cell_type, x, y, cell_id, label,
    label_map         : arrays returned by utils.preprocess.process_csv
    K                 : number of nearest neighbours per anchor
    ncell             : number of the K neighbours randomly kept (<= K)
    random_ctrl       : add a "background" class of random non-anchor
                        neighbourhoods (ignored when anchor == "BG").
                        Recommended for anchored runs: it is the only way the
                        package tests anchor specificity.
    task              : only "classification" is implemented
    nruns             : number of independent training runs
    dendrogram_cutoff : cosine-distance cut for clustering filters
    n_val_folds       : number of patient-level train/validation folds
    maxpool_percentages: pooling sizes, % of ncell, cycled through by run
    nfilter_selection : candidate numbers of filters, drawn per run
    filter_threshold  : quantile above which neighbourhoods are "selected"
                        (0.9 = top 10%)
    dropout, dropout_p, bg_sets, batch_size, lr, l2, epochs, num_workers,
    early_stopping, patience : training settings (lr default here is 0.1,
                        whereas fit() defaults to 0.01)
    seed              : global seed; also decides which patients form the test set
    save_path         : directory for the timestamped results folder

    Returns
    -------
    None. All outputs are written to disk under
    <save_path>/results_Anchor<anchor>_K<K>_<date>_<time>/.
    """
    # Set seed
    set_seed(seed)

    # Create dataset from processed csv
    dataset = CIMADataset(
        anchor = anchor,
        image = image_id,
        pat = sample_id,
        intensity = intensity,
        ct = cell_type,
        x = x,
        y = y,
        cellid = cell_id,
        K  = K,
        ncell = ncell,
        label = label,
        random_ctrl = random_ctrl,
        task = task,
        bg_sets=bg_sets,
        seed = seed,
    )

    # Fit the CIMA model from this dataset
    results, loss, val_loss, ba, val_ba, test_samples, train_folds, res_save_path = fit(dataset,
        anchor = anchor,
        K = K,
        ncell = ncell,
        genes = genes,
        n_val_folds=n_val_folds,
        nruns=nruns,
        # NOTE: fit() uses dataset.random_ctrl (the effective value) for the
        # background logic, since the dataset overrides it for "BG" (FIX-4).
        background=random_ctrl,
        batch_size = batch_size,
        lr = lr,
        epochs=epochs,
        dendrogram_cutoff = dendrogram_cutoff,
        maxpool_percentages = maxpool_percentages,
        nfilter_selection = nfilter_selection,
        dropout = dropout,
        dropout_p = dropout_p,
        l2 = l2,
        early_stopping = early_stopping,
        patience = patience,
        num_workers = num_workers,
        save_loc = save_path,
        seed=seed)
    
    # Calculate cell filter response - results is the final metadata dict
    # (list of the up-to-4 consensus filters chosen for reporting)
    fig_save_path = s3cima_plot.plot_filter_weights(results,
                                    show=False, 
                                    save_path=res_save_path)   
    res_test = s3cima_plot.get_high_response_cells_test(results, 
                                            test_samples, 
                                            genes, 
                                            filter_threshold=filter_threshold)
    res_train = s3cima_plot.get_high_response_cells_train(results, 
                                              train_folds, 
                                              genes, 
                                              filter_threshold=filter_threshold)
    
    # Calculate enrichment and plots.
    # test/  : held-out patients (also used to rank which filters are
    #          reported, so read them as a generalisation check of the chosen
    #          filters rather than an untouched estimate)
    # train/ : all non-test patients
    test_fig_save_path = s3cima_plot.save_stats(res_test,
                                    x, y, cell_id, cell_type, sample_id, label, label_map,
                                    save_path = fig_save_path,
                                    test = True)
    train_fig_save_path = s3cima_plot.save_stats(res_train,
                                    x, y, cell_id, cell_type, sample_id, label, label_map,
                                    save_path = fig_save_path,
                                    test = False)
    

    s3cima_plot.enrichment_summary(test_fig_save_path,
                                   cell_type, sample_id,
                                   test=True)
    s3cima_plot.enrichment_summary(train_fig_save_path,
                                   cell_type, sample_id,
                                   test=False)

