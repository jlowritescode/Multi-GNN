import torch
import tqdm
from sklearn.metrics import f1_score
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero, save_model, load_model
from models import GINe, PNA, GATe, RGCN
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
import wandb
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_curve
)
def save_pattern_recall_reports(
    metrics,
    args,
    data_config,
    output_dir
):
    """
    Calculates laundering detection recall separately
    for each known AML pattern.

    Produces:
      - paper-style 6-pattern + NONE results
      - all-8-pattern results
      - results at thresholds 0.05 ... 0.95
      - 0.50 threshold summary
      - pattern recall graph
    """

    y_true = np.asarray(
        metrics["y_true"]
    )

    y_score = np.asarray(
        metrics["y_score"]
    )

    edge_ids = np.asarray(
        metrics["edge_ids"],
        dtype=int
    )

    # ==========================================================
    # LOAD PATTERNS FROM FORMATTED TRANSACTIONS
    # ==========================================================

    transaction_file = (
        Path(data_config["paths"]["aml_data"])
        / args.data
        / "formatted_transactions.csv"
    )

    transaction_df = pd.read_csv(
        transaction_file,
        low_memory=False
    )

    if "Pattern" not in transaction_df.columns:
        raise ValueError(
            "\nPattern column was not found in "
            "formatted_transactions.csv.\n"
            "Rerun format_kaggle_files.py after adding "
            "the pattern-label preprocessing code."
        )

    # edge_ids are row positions in te_data /
    # formatted_transactions.csv
    test_patterns = (
        transaction_df
        .iloc[edge_ids]["Pattern"]
        .fillna("NONE")
        .astype(str)
        .str.upper()
        .to_numpy()
    )

    if not (
        len(test_patterns)
        == len(y_true)
        == len(y_score)
    ):
        raise ValueError(
            "Pattern labels and model predictions "
            "are not aligned."
        )

    # ==========================================================
    # PATTERN DEFINITIONS
    # ==========================================================

    paper_patterns = [
        "FAN-IN",
        "FAN-OUT",
        "CYCLE",
        "SCATTER-GATHER",
        "GATHER-SCATTER",
        "BIPARTITE",
        "NONE"
    ]

    all_patterns = [
        "FAN-IN",
        "FAN-OUT",
        "CYCLE",
        "SCATTER-GATHER",
        "GATHER-SCATTER",
        "BIPARTITE",
        "STACK",
        "RANDOM",
        "NONE"
    ]

    display_names = {
        "FAN-IN": "Fan-in",
        "FAN-OUT": "Fan-out",
        "CYCLE": "Cycle",
        "SCATTER-GATHER": "Scatter-Gather",
        "GATHER-SCATTER": "Gather-Scatter",
        "BIPARTITE": "Bipartite",
        "STACK": "Stack",
        "RANDOM": "Random",
        "NONE": "None"
    }

    thresholds = np.arange(
        0.05,
        1.00,
        0.05
    )

    # ==========================================================
    # HELPER
    # ==========================================================

    def calculate_table(pattern_list):

        rows = []

        for threshold in thresholds:

            predictions = (
                y_score >= threshold
            ).astype(int)

            for pattern in pattern_list:

                # Only evaluate actual laundering transactions
                # belonging to this pattern.
                mask = (
                    (y_true == 1)
                    & (test_patterns == pattern)
                )

                n_transactions = int(
                    mask.sum()
                )

                if n_transactions == 0:

                    detected = 0
                    missed = 0
                    recall = np.nan

                else:

                    detected = int(
                        predictions[mask].sum()
                    )

                    missed = (
                        n_transactions
                        - detected
                    )

                    recall = (
                        detected
                        / n_transactions
                    )

                rows.append({
                    "threshold": round(
                        float(threshold),
                        2
                    ),
                    "pattern": display_names[pattern],
                    "n_test_laundering_transactions":
                        n_transactions,
                    "detected": detected,
                    "missed": missed,
                    "recall": recall
                })

        return pd.DataFrame(rows)

    # ==========================================================
    # PAPER-STYLE RESULTS
    # ==========================================================

    paper_df = calculate_table(
        paper_patterns
    )

    paper_path = (
        output_dir
        / "pattern_recall_by_threshold.csv"
    )

    paper_df.to_csv(
        paper_path,
        index=False
    )

    # ==========================================================
    # EXPANDED ALL-8-PATTERN RESULTS
    # ==========================================================

    all_df = calculate_table(
        all_patterns
    )

    all_path = (
        output_dir
        / "pattern_recall_by_threshold_all_patterns.csv"
    )

    all_df.to_csv(
        all_path,
        index=False
    )

    # ==========================================================
    # 0.50 SUMMARY
    # ==========================================================

    threshold_050_df = paper_df[
        np.isclose(
            paper_df["threshold"],
            0.50
        )
    ].copy()

    threshold_050_path = (
        output_dir
        / "pattern_recall_at_0.50.csv"
    )

    threshold_050_df.to_csv(
        threshold_050_path,
        index=False
    )

    print("\n")
    print("=" * 80)
    print("TEST RECALL BY MONEY-LAUNDERING PATTERN @ THRESHOLD 0.50")
    print("=" * 80)

    print(
        threshold_050_df[
            [
                "pattern",
                "n_test_laundering_transactions",
                "detected",
                "missed",
                "recall"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("=" * 80)

    # ==========================================================
    # GRAPH
    # ==========================================================

    plot_df = paper_df.pivot(
        index="threshold",
        columns="pattern",
        values="recall"
    )

    plt.figure(
        figsize=(10, 7)
    )

    for column in plot_df.columns:

        plt.plot(
            plot_df.index,
            plot_df[column],
            marker="o",
            label=column
        )

    plt.xlabel(
        "Laundering Classification Threshold"
    )

    plt.ylabel(
        "Recall"
    )

    plt.title(
        "Test Recall by Money-Laundering Pattern"
    )

    plt.xticks(
        thresholds,
        rotation=45
    )

    plt.ylim(
        0,
        1.05
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plot_path = (
        output_dir
        / "pattern_recall_by_threshold.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ==========================================================
    # WANDB
    # ==========================================================

    if not args.testing:

        wandb.log({
            "final_test/pattern_recall_plot":
                wandb.Image(
                    str(plot_path)
                )
        })

        wandb.log({
            "final_test/pattern_recall_table":
                wandb.Table(
                    dataframe=threshold_050_df
                )
        })

    print("\nPATTERN RESULTS SAVED:")

    print(
        f"  Paper-style threshold results: "
        f"{paper_path}"
    )

    print(
        f"  All pattern results: "
        f"{all_path}"
    )

    print(
        f"  Pattern recall @ 0.50: "
        f"{threshold_050_path}"
    )

    print(
        f"  Pattern graph: "
        f"{plot_path}"
    )

    return paper_df
def final_test_report(
    metrics,
    args,
    data_config):
    """
    Generate and save final test-set classification results.

    Outputs:
      1. Final default-threshold metrics
      2. Metrics at thresholds 0.05 through 0.95
      3. Confusion counts at every threshold
      4. CSV threshold table
      5. Continuous precision-recall curve
      6. Precision/Recall/F1 vs threshold graph
      7. W&B results
    """

    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    y_score = metrics["y_score"]

    # ============================================================
    # DEFAULT MODEL RESULTS
    # ============================================================

    default_precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    default_recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    default_f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    average_precision = average_precision_score(
        y_true,
        y_score
    )

    print("\n")
    print("=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    print(
        f"Precision:          "
        f"{default_precision:.4f}"
    )

    print(
        f"Recall:             "
        f"{default_recall:.4f}"
    )

    print(
        f"F1 Score:           "
        f"{default_f1:.4f}"
    )

    print(
        f"Average Precision:  "
        f"{average_precision:.4f}"
    )

    print("=" * 65)

    logging.info(
        "FINAL TEST RESULTS -- "
        f"Precision: {default_precision:.4f}, "
        f"Recall: {default_recall:.4f}, "
        f"F1: {default_f1:.4f}, "
        f"Average Precision: {average_precision:.4f}"
    )

    # ============================================================
    # CREATE UNIQUE DIRECTORY FOR THIS RUN
    # ============================================================

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    adaptations = []

    if args.emlps:
        adaptations.append("emlps")

    if args.reverse_mp:
        adaptations.append("reversemp")

    if args.ego:
        adaptations.append("ego")

    if args.ports:
        adaptations.append("ports")

    if args.tds:
        adaptations.append("tds")

    model_name = f"{args.data}_{args.model}"

    if adaptations:
        model_name += "_" + "_".join(adaptations)

    run_name = (
        f"{model_name}"
        f"_seed{args.seed}"
        f"_{timestamp}"
    )

    output_dir = Path("results") / run_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    pattern_results = save_pattern_recall_reports(
        metrics,
        args,
        data_config,
        output_dir
    )
    # ============================================================
    # FIXED THRESHOLDS: .05 INCREMENTS
    # ============================================================

    thresholds = np.arange(
        0.05,
        1.00,
        0.05
    )

    threshold_results = []

    for threshold in thresholds:

        # Anything >= threshold becomes laundering
        threshold_pred = (
            y_score >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            threshold_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            threshold_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            threshold_pred,
            zero_division=0
        )

        # Confusion-matrix components
        tp = int(
            np.sum(
                (y_true == 1)
                & (threshold_pred == 1)
            )
        )

        fp = int(
            np.sum(
                (y_true == 0)
                & (threshold_pred == 1)
            )
        )

        tn = int(
            np.sum(
                (y_true == 0)
                & (threshold_pred == 0)
            )
        )

        fn = int(
            np.sum(
                (y_true == 1)
                & (threshold_pred == 0)
            )
        )

        predicted_laundering = (
            tp + fp
        )

        actual_laundering = (
            tp + fn
        )

        threshold_results.append({
            "threshold": round(
                float(threshold),
                2
            ),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "predicted_laundering": predicted_laundering,
            "actual_laundering": actual_laundering
        })

    threshold_df = pd.DataFrame(
        threshold_results
    )

    # ============================================================
    # PRINT THRESHOLD TABLE
    # ============================================================

    print("\n")
    print("=" * 90)
    print("TEST RESULTS BY CLASSIFICATION THRESHOLD")
    print("=" * 90)

    print(
        threshold_df[
            [
                "threshold",
                "precision",
                "recall",
                "f1",
                "true_positives",
                "false_positives",
                "false_negatives"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("=" * 90)

    # ============================================================
    # SAVE THRESHOLD CSV
    # ============================================================

    threshold_csv_path = (
        output_dir
        / "test_threshold_metrics.csv"
    )

    threshold_df.to_csv(
        threshold_csv_path,
        index=False
    )

    # ============================================================
    # SAVE DEFAULT FINAL METRICS
    # ============================================================

    final_metrics_df = pd.DataFrame(
        [{
            "precision": default_precision,
            "recall": default_recall,
            "f1": default_f1,
            "average_precision": average_precision
        }]
    )

    final_metrics_path = (
        output_dir
        / "final_test_metrics.csv"
    )

    final_metrics_df.to_csv(
        final_metrics_path,
        index=False
    )

    # ============================================================
    # CONTINUOUS PRECISION-RECALL CURVE
    # ============================================================

    (
        curve_precision,
        curve_recall,
        curve_thresholds
    ) = precision_recall_curve(
        y_true,
        y_score
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        curve_recall,
        curve_precision,
        linewidth=2
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Test Precision-Recall Curve\n"
        f"Average Precision = "
        f"{average_precision:.4f}"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    pr_curve_path = (
        output_dir
        / "precision_recall_curve.png"
    )

    plt.savefig(
        pr_curve_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ============================================================
    # FIXED THRESHOLD METRICS GRAPH
    # ============================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        threshold_df["threshold"],
        threshold_df["precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        threshold_df["threshold"],
        threshold_df["recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        threshold_df["threshold"],
        threshold_df["f1"],
        marker="o",
        label="F1"
    )

    plt.xlabel(
        "Laundering Classification Threshold"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Test Precision, Recall, and F1 "
        "by Classification Threshold"
    )

    plt.xticks(
        thresholds,
        rotation=45
    )

    plt.ylim(
        0,
        1.05
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    threshold_plot_path = (
        output_dir
        / "precision_recall_f1_by_threshold.png"
    )

    plt.savefig(
        threshold_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ============================================================
    # WANDB
    # ============================================================

    if not args.testing:

        wandb.log({
            "final_test/precision":
                default_precision,

            "final_test/recall":
                default_recall,

            "final_test/f1":
                default_f1,

            "final_test/average_precision":
                average_precision,

            "final_test/pr_curve":
                wandb.Image(
                    str(pr_curve_path)
                ),

            "final_test/threshold_metrics":
                wandb.Image(
                    str(threshold_plot_path)
                )
        })

        # Upload threshold table to W&B
        wandb.log({
            "final_test/threshold_table":
                wandb.Table(
                    dataframe=threshold_df
                )
        })

    # ============================================================
    # PRINT SAVED FILE LOCATIONS
    # ============================================================

    print("\nFINAL TEST FILES SAVED:")

    print(
        f"  Metrics: "
        f"{final_metrics_path}"
    )

    print(
        f"  Threshold table: "
        f"{threshold_csv_path}"
    )

    print(
        f"  PR curve: "
        f"{pr_curve_path}"
    )

    print(
        f"  Threshold graph: "
        f"{threshold_plot_path}"
    )

    print()

def train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    best_val_f1 = 0
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            #remove the unique edge id from the edge features, as it's no longer needed
            batch.edge_attr = batch.edge_attr[:, 1:]

            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = out[mask]
            ground_truth = batch.y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(ground_truth)
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        wandb.log({"f1/train": f1}, step=epoch)
        logging.info(f'Train F1: {f1:.4f}')

        #evaluate
        val_f1 = evaluate_homo(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_homo(te_loader, te_inds, model, te_data, device, args)

        wandb.log({"f1/validation": val_f1}, step=epoch)
        wandb.log({"f1/test": te_f1}, step=epoch)
        logging.info(f'Validation F1: {val_f1:.4f}')
        logging.info(f'Test F1: {te_f1:.4f}')

        if epoch == 0:
            wandb.log({"best_test_f1": te_f1}, step=epoch)
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    final_metrics = evaluate_homo(
        te_loader,
        te_inds,
        model,
        te_data,
        device,
        args,
        return_details=True
    )

    final_test_report(
        final_metrics,
        args,
        data_config
    )

    return model

def train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    best_val_f1 = 0
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)
            
            #remove the unique edge id from the edge features, as it's no longer needed
            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            out = out[('node', 'to', 'node')]
            pred = out[mask]
            ground_truth = batch['node', 'to', 'node'].y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(batch['node', 'to', 'node'].y[mask])
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            
        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        wandb.log({"f1/train": f1}, step=epoch)
        logging.info(f'Train F1: {f1:.4f}')

        #evaluate
        val_f1 = evaluate_hetero(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_hetero(te_loader, te_inds, model, te_data, device, args)

        wandb.log({"f1/validation": val_f1}, step=epoch)
        wandb.log({"f1/test": te_f1}, step=epoch)
        logging.info(f'Validation F1: {val_f1:.4f}')
        logging.info(f'Test F1: {te_f1:.4f}')

        if epoch == 0:
            wandb.log({"best_test_f1": te_f1}, step=epoch)
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
        
    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    final_metrics = evaluate_hetero(
        te_loader,
        te_inds,
        model,
        te_data,
        device,
        args,
        return_details=True
    )

    final_test_report(
        final_metrics,
        args,
        data_config
    )

    return model

def get_model(sample_batch, config, args):
    n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    e_dim = (sample_batch.edge_attr.shape[1] - 1) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)

    if args.model == "gin":
        model = GINe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), residual=False, edge_updates=args.emlps, edge_dim=e_dim, 
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "gat":
        model = GATe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), n_heads=round(config.n_heads), 
                edge_updates=args.emlps, edge_dim=e_dim,
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "pna":
        if not isinstance(sample_batch, HeteroData):
            d = degree(sample_batch.edge_index[1], dtype=torch.long)
        else:
            index = torch.cat((sample_batch['node', 'to', 'node'].edge_index[1], sample_batch['node', 'rev_to', 'node'].edge_index[1]), 0)
            d = degree(index, dtype=torch.long)
        deg = torch.bincount(d, minlength=1)
        model = PNA(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=e_dim,
            dropout=config.dropout, deg=deg, final_dropout=config.final_dropout
            )
    elif config.model == "rgcn":
        model = RGCN(
            num_features=n_feats, edge_dim=e_dim, num_relations=8, num_gnn_layers=round(config.n_gnn_layers),
            n_classes=2, n_hidden=round(config.n_hidden),
            edge_update=args.emlps, dropout=config.dropout, final_dropout=config.final_dropout, n_bases=None #(maybe)
        )
    
    return model

def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #define a model config dictionary and wandb logging at the same time
    wandb.init(
        mode="disabled" if args.testing else "online",
        project="your_proj_name", #replace this with your wandb project name if you want to use wandb logging

        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
        }
    )

    config = wandb.config

    #set the transform if ego ids should be used
    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    #add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    #get the model
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)

    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')
    
    if args.finetune:
        model, optimizer = load_model(model, device, args, config, data_config)
    else:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    sample_batch.to(device)
    sample_x = sample_batch.x if not isinstance(sample_batch, HeteroData) else sample_batch.x_dict
    sample_edge_index = sample_batch.edge_index if not isinstance(sample_batch, HeteroData) else sample_batch.edge_index_dict
    if isinstance(sample_batch, HeteroData):
        sample_batch['node', 'to', 'node'].edge_attr = sample_batch['node', 'to', 'node'].edge_attr[:, 1:]
        sample_batch['node', 'rev_to', 'node'].edge_attr = sample_batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
    else:
        sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    sample_edge_attr = sample_batch.edge_attr if not isinstance(sample_batch, HeteroData) else sample_batch.edge_attr_dict
    logging.info(summary(model, sample_x, sample_edge_index, sample_edge_attr))
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))

    if args.reverse_mp:
        model = train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    else:
        model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    
    wandb.finish()