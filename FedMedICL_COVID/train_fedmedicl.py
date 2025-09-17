#!/usr/bin/env python
"""
train_fedmedicl.py

This script trains the enhanced FedMedICL pipeline using our novel FedDG approach with a TabTransformer model.
The dataset is located at /content/COVID-19_Case_Surveillance_Public_Use_Data.csv.
It splits the data into multiple clients (simulating client IDs if necessary),
and performs federated training with dual batch normalization and BN adapter for domain adaptation.
After training, it computes extended metrics (accuracy, precision, recall, F1, ROC-AUC, average precision, NDCG)
and saves a ROC curve plot along with all training artifacts in the experiment folder.
"""

import torch.multiprocessing as mp
try:
    mp.set_start_method('forkserver', force=True)
except RuntimeError:
    pass

import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import custom modules
from dataset.CovidSurveillanceDataset import CovidSurveillanceDataset, create_federated_datasets
from model.TabTransformer import TabTransformer
from training.federated_trainer import federated_train_FedDG
from reporting.metrics import evaluate_model, compute_classification_metrics
from utils.logger import init_logger
from utils.saver import create_experiment_folder, save_config, save_model, save_metrics, save_plot

def main():
    parser = argparse.ArgumentParser(description="FedMedICL Training with FedDG & TabTransformer")
    parser.add_argument("--dataset", type=str, default="/content/COVID-19_Case_Surveillance_Public_Use_Data.csv",
                        help="Path to the CSV dataset")
    parser.add_argument("--model", type=str, default="tabtransformer",
                        help="Model architecture to use (e.g., tabtransformer)")
    parser.add_argument("--algorithm", type=str, default="FedDG",
                        help="Federated training algorithm to use (e.g., FedDG, FedAvg, ERM)")
    parser.add_argument("--num_clients", type=int, default=5,
                        help="Number of federated clients")
    parser.add_argument("--num_tasks", type=int, default=4,
                        help="Number of sequential tasks (reserved for future extensions)")
    parser.add_argument("--rounds", type=int, default=50,
                        help="Communication rounds for main training")
    parser.add_argument("--adapter_rounds", type=int, default=5,
                        help="Rounds for BN adapter fine-tuning")
    parser.add_argument("--emb_dim", type=int, default=64,
                        help="Embedding dimension for TabTransformer")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension for TabTransformer")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of transformer layers in TabTransformer")
    parser.add_argument("--transformer_heads", type=int, default=8,
                        help="Number of heads in the transformer")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for the transformer encoder")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for optimizer")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["none", "step", "cosine"],
                        help="Learning rate scheduler type")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed-precision (fp16) training")
    parser.add_argument("--output_dir", type=str, default="./runs",
                        help="Base directory for logs, checkpoints, and outputs")
    args = parser.parse_args()

    # ---------- Experiment Folder Setup ----------
    experiment_folder = create_experiment_folder("./FedMedICL_COVID/experiments", args.algorithm, args.rounds)
    args.output_dir = experiment_folder

    # Initialize logger and save configuration
    cfg = vars(args).copy()
    log = init_logger(cfg, out_dir=experiment_folder)
    save_config(cfg, os.path.join(experiment_folder, "args.txt"))

    # -------------------- Data & Feature Setup --------------------
    data_path = args.dataset
    cat_features = ["age_group", "sex", "Race and ethnicity (combined)", "medcond_yn"]
    num_features = []  # No numerical features in this example.
    label_col = "death_yn"
    
    client_ids = [str(i) for i in range(args.num_clients)]
    federated_datasets = create_federated_datasets(data_path, client_ids, cat_features, num_features, label_col)
    
    federated_data = {}
    for cid, dataset in federated_datasets.items():
        dl = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=args.num_workers,
                        persistent_workers=True,
                        timeout=60,
                        prefetch_factor=2)
        federated_data[cid] = dl

    # -------------------- Model Initialization --------------------
    if args.model.lower() == "tabtransformer":
        df = pd.read_csv(data_path, low_memory=False)
        num_categories = [len(df[col].astype("category").cat.categories) for col in cat_features]
        model = TabTransformer(num_categories=num_categories,
                               num_numeric=len(num_features),
                               emb_dim=args.emb_dim,
                               transformer_heads=args.transformer_heads,
                               transformer_layers=args.n_layers,
                               hidden_dim=args.hidden_dim,
                               num_classes=2,
                               dropout=args.dropout)
    else:
        raise ValueError("Unsupported model architecture. Use 'tabtransformer'.")

    # -------------------- Federated Training --------------------
    # In train_fedmedicl.py, inside the main function:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on device: {device}")
    algo = args.algorithm.lower()
    if algo == "feddg":
        log.info("Starting FedDG training...")
        trained_model = federated_train_FedDG(global_model=model,
                                              clients_datasets=federated_data,
                                              rounds=args.rounds,
                                              adapter_rounds=args.adapter_rounds,
                                              lr=args.lr,
                                              adapter_lr=args.lr / 2,
                                              weight_decay=args.weight_decay,
                                              amp=args.amp,
                                              device=device,
                                              log=log)
    elif algo == "feddghybrid":
        log.info("Starting FedDGHybrid training...")
        from training.algorithms import federated_train_FedDGHybrid
        trained_model = federated_train_FedDGHybrid(global_model=model,
                                                    clients_datasets=federated_data,
                                                    rounds=args.rounds,
                                                    adapter_rounds=args.adapter_rounds,
                                                    lr=args.lr,
                                                    adapter_lr=args.lr / 2,
                                                    weight_decay=args.weight_decay,
                                                    amp=args.amp,
                                                    device=device,
                                                    log=log)
    elif algo == "fedcb":
        log.info("Starting FedCB training...")
        from training.algorithms import federated_train_FedCB
        trained_model = federated_train_FedCB(global_model=model,
                                              clients_datasets=federated_data,
                                              rounds=args.rounds,
                                              lr=args.lr,
                                              weight_decay=args.weight_decay,
                                              amp=args.amp,
                                              device=device,
                                              log=log)
    elif algo == "fedavg":
        log.info("Starting FedAvg training...")
        from training.algorithms import federated_train_FedAvg
        trained_model = federated_train_FedAvg(global_model=model,
                                               clients_datasets=federated_data,
                                               rounds=args.rounds,
                                               lr=args.lr,
                                               weight_decay=args.weight_decay,
                                               amp=args.amp,
                                               device=device,
                                               log=log)
    elif algo == "erm":
        log.info("Starting ERM training (centralized baseline)...")
        from torch.utils.data import ConcatDataset
        from training.algorithms import train_erm
        merged_dataset = ConcatDataset([ds for ds in federated_datasets.values()])
        merged_loader = DataLoader(merged_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=args.num_workers)
        trained_model = train_erm(global_model=model,
                                  merged_loader=merged_loader,
                                  epochs=10,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amp=args.amp,
                                  device=device,
                                  log=log)
    elif algo == "fedaim":
      log.info("Starting FedAIM training...")
      from training.algorithms import federated_train_FedAIM
      trained_model = federated_train_FedAIM(
          global_model=model,
          clients_datasets=federated_data,
          rounds=args.rounds,
          adapter_rounds=args.adapter_rounds,
          lr=args.lr,
          adapter_lr=args.lr / 2,
          weight_decay=args.weight_decay,
          amp=args.amp,
          device=device,
          log=log,
          lambda_consist=0.7,  # adjust as needed
          beta_kd=0.3,        # adjust as needed
          meta_eta=0.15,       # meta-learning rate for adaptation of the positive class weight
          target_recall=0.7   # desired target recall for the positive class
      )
    else:
        raise ValueError("Unsupported algorithm specified.")


    # -------------------- Save Final Model --------------------
    model_save_path = os.path.join(experiment_folder, "final_model.pth")
    save_model(trained_model, model_save_path)
    log.info(f"Final model saved to: {model_save_path}")

    # -------------------- Extended Evaluation and Plotting --------------------
    all_metrics = {}
    for cid, dl in federated_data.items():
        accuracy, preds, labels, probs = evaluate_model(trained_model, dl, device=device, use_adapter=True)
        ext_metrics = compute_classification_metrics(labels, probs)
        # Add basic metrics and sample count
        ext_metrics["accuracy"] = accuracy
        ext_metrics["num_samples"] = len(dl.dataset)
        all_metrics[cid] = ext_metrics
        log.info(f"Client {cid} Extended Metrics: {ext_metrics}")
        
        # Plot ROC curve for each client and save via saver.py function save_plot
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve for Client {cid}')
        ax.legend(loc="lower right")
        roc_plot_path = os.path.join(experiment_folder, f"roc_curve_client_{cid}.png")
        save_plot(fig, roc_plot_path)
        log.info(f"ROC curve saved for client {cid} to: {roc_plot_path}")

    metrics_save_path = os.path.join(experiment_folder, "metrics.json")
    save_metrics(all_metrics, metrics_save_path)
    log.info(f"Extended metrics saved to: {metrics_save_path}")

if __name__ == "__main__":
    main()
