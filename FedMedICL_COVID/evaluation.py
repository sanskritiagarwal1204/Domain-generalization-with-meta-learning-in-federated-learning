#!/usr/bin/env python
"""
evaluation.py

This script evaluates a trained FedMedICL model (e.g., a checkpoint saved from training)
on a test dataset. It computes extended classification metrics and generates a ROC curve plot.
"""

import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset.CovidSurveillanceDataset import CovidSurveillanceDataset
from model.TabTransformer import TabTransformer
from reporting.metrics import evaluate_model, compute_classification_metrics
from utils.logger import init_logger
from utils.saver import save_metrics

def plot_roc_curve(y_true, y_prob, save_path):
    """
    Generate and save a ROC curve plot.
    
    Args:
      y_true: Ground-truth labels.
      y_prob: Predicted probabilities for the positive class.
      save_path: File path where the plot should be saved.
    """
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluation Script for FedMedICL Model")
    parser.add_argument("--dataset", type=str, default="/content/COVID-19_Case_Surveillance_Public_Use_Data.csv",
                        help="Path to the CSV dataset")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the saved model checkpoint (.pth)")
    parser.add_argument("--model", type=str, default="tabtransformer", help="Model architecture")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Directory to save evaluation outputs")
    parser.add_argument("--use_adapter", action="store_true", help="Use adapter-based forward pass")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    log = init_logger(vars(args), args.output_dir)
    
    # Data & Feature Setup
    data_path = args.dataset
    cat_features = ["age_group", "sex", "Race and ethnicity (combined)", "medcond_yn"]
    num_features = []  # No numerical features in this example.
    label_col = "death_yn"
    
    # Create the dataset (note: we use the variable name 'dataset' consistently)
    dataset = CovidSurveillanceDataset(data_path, cat_features=cat_features, num_features=num_features, label_column=label_col)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    
    # Model Initialization
    df = pd.read_csv(data_path, low_memory=False)
    num_categories = [len(df[col].astype("category").cat.categories) for col in cat_features]
    model = TabTransformer(num_categories=num_categories,
                           num_numeric=len(num_features),
                           emb_dim=256,
                           transformer_heads=8,
                           transformer_layers=4,
                           hidden_dim=512,
                           num_classes=2,
                           dropout=0.1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        log.info(f"Loaded checkpoint from {args.checkpoint}")
    else:
        log.error("Checkpoint file does not exist.")
        return
    
    # Evaluate the model
    accuracy, preds, labels, probs = evaluate_model(model, loader, device=device, use_adapter=args.use_adapter)
    log.info(f"Overall Accuracy: {accuracy*100:.2f}%")
    
    extended_metrics = compute_classification_metrics(labels, probs, probs)
    log.info(f"Extended Metrics: {extended_metrics}")
    
    metrics_save_path = os.path.join(args.output_dir, "eval_metrics.json")
    save_metrics(extended_metrics, metrics_save_path)
    log.info(f"Metrics saved to: {metrics_save_path}")
    
    # Generate and save ROC curve plot
    roc_plot_path = os.path.join(args.output_dir, "roc_curve.png")
    plot_roc_curve(labels, probs, roc_plot_path)
    log.info(f"ROC curve plot saved to: {roc_plot_path}")

if __name__ == "__main__":
    main()
