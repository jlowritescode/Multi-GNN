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

def final_test_report(metrics, args):
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
        args
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
        args
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