#!/usr/bin/env python
import sys
import os
import argparse
import yaml
import torch
import time
import numpy as np
from datetime import timedelta

# Append repository root to ensure submodules are importable.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import project modules.
from dataset.COVIDDataset import load_and_split_data
from model.TabularModel import TabularModel
from training.client import Client
from training.server import Server
from training import algorithms, metrics

# ========= Argument Parsing ==========
parser = argparse.ArgumentParser(
    description="Federated COVID-19 Training Pipeline with FL & ERM Baselines and Imbalance Handling"
)
parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
parser.add_argument('--dataset', type=str, default=None, help='Path to COVID CSV dataset')
parser.add_argument('--algorithm', type=str, default=None,
                    choices=['FedAvg', 'FedAvgTemporal', 'FedCB', 'ERM'],
                    help='Training algorithm: choose FedAvg, FedAvgTemporal, FedCB (Fed Class-Balanced), or ERM (centralized baseline)')
parser.add_argument('--num_clients', type=int, default=None, help='Number of clients')
parser.add_argument('--num_tasks', type=int, default=None, help='Number of temporal tasks')
parser.add_argument('--no_holdout', action='store_true', help='Train on all tasks without holding out the last task for evaluation')
parser.add_argument('--rounds', type=int, default=None, help='Rounds per task')
parser.add_argument('--local_epochs', type=int, default=None, help='Local epochs per round (for FL methods)')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training')
parser.add_argument('--lr', type=float, default=None, help='Learning rate')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory for logs and plots')
parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                    help='Device to run training on (cpu or cuda)')
args = parser.parse_args()

# ========= Load or Build Configuration ==========
config = {}
if args.config:
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
else:
    config['dataset'] = {
        'path': args.dataset,
        'num_clients': args.num_clients or 10,
        'num_tasks': args.num_tasks or 4,
        'holdout_last_task': False if args.no_holdout else True
    }
    config['model'] = {
        'name': "TabularFFN",
        'embed_dim': 8,
        'hidden_dim': 32,
        'num_layers': 2,
        'dropout': 0.0
    }
    config['training'] = {
        'algorithm': args.algorithm or "FedAvg",
        'rounds_per_task': args.rounds or 50,
        'local_epochs': args.local_epochs or 1,
        'local_batch_size': args.batch_size or 32,
        'learning_rate': args.lr or 0.001,
        'seed': args.seed or 42
    }
    config['logging'] = {
        'output_dir': args.output_dir or "./fed_output",
        'use_wandb': bool(args.use_wandb)
    }

# ========= Device Setup ==========
if args.device.lower() == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("Running on device:", device)

# ========= Print Configuration ==========
separator = "=" * 60
print(separator)
print("Federated COVID-19 Training Pipeline Configuration:")
print(yaml.dump(config, default_flow_style=False))
print(separator)

# ========= Logging Setup ==========
log_cfg = config.get('logging', {})
output_dir = log_cfg.get('output_dir', './fed_output')
os.makedirs(output_dir, exist_ok=True)
log_file_path = os.path.join(output_dir, "training.log")
log_file = open(log_file_path, "w")
log_file.write(separator + "\n")
log_file.write("Federated COVID-19 Training Pipeline Configuration:\n")
log_file.write(yaml.dump(config, default_flow_style=False) + "\n")
log_file.write(separator + "\n")

# ========= Set Random Seed ==========
seed = config['training'].get('seed')
if seed is not None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
log_file.write(f"Random seed set to: {seed}\n")

# ========= Optional: Initialize WandB Early ==========
if log_cfg.get('use_wandb', False):
    try:
        import wandb
        wandb.init(project="fed_covid", config=config)
        print("Weights & Biases logging enabled (initialized early).")
        log_file.write("Weights & Biases logging enabled (initialized early).\n")
    except Exception as e:
        print("WandB initialization failed:", e)
        log_file.write(f"WandB initialization failed: {e}\n")

# ========= Data Loading ==========
data_cfg = config['dataset']
csv_path = data_cfg['path']
print(f"Starting data load from: {csv_path}")
log_file.write(f"Starting data load from: {csv_path}\n")
data_start_time = time.time()
clients_train_datasets, clients_test_datasets, feature_maps = load_and_split_data(
    csv_path,
    num_clients=data_cfg['num_clients'],
    num_tasks=data_cfg['num_tasks'],
    holdout_last_task=data_cfg.get('holdout_last_task', True)
)
data_end_time = time.time()
data_elapsed = data_end_time - data_start_time
print(f"Dataset loaded in {data_elapsed:.2f} seconds.")
log_file.write(f"Dataset loaded in {data_elapsed:.2f} seconds.\n")

# ========= Global Model Initialization ==========
model_cfg = config['model']
global_model = TabularModel(feature_maps,
                            embed_dim=model_cfg.get('embed_dim', 8),
                            hidden_dim=model_cfg.get('hidden_dim', 32),
                            num_layers=model_cfg.get('num_layers', 2),
                            dropout=model_cfg.get('dropout', 0.0))
print("Global model initialized.")
log_file.write("Global model initialized.\n")

# ========= Client Creation (for Federated Modes) ==========
clients = []
for i, train_datasets in enumerate(clients_train_datasets):
    client_obj = Client(client_id=i,
                        train_datasets=train_datasets,
                        test_dataset=clients_test_datasets[i] if i < len(clients_test_datasets) else None)
    clients.append(client_obj)
print(f"Created {len(clients)} client(s) for FL mode.")
log_file.write(f"Created {len(clients)} client(s) for FL mode.\n")

# ========= Server Setup ==========
server = Server(global_model, clients, device=device)
log_file.write(f"Using device: {device}\n")

# ========= Training Parameters ==========
train_cfg = config['training']
algorithm = train_cfg.get('algorithm', 'FedAvg')
rounds_per_task = train_cfg['rounds_per_task']
local_epochs = train_cfg['local_epochs']
batch_size = train_cfg['local_batch_size']
lr = train_cfg['learning_rate']

print("\nStarting training using algorithm:", algorithm)
log_file.write(f"\nStarting training using algorithm: {algorithm}\n")

# ========= Run Training Based on Algorithm Choice ==========
if algorithm.lower() == "erm":
    print("Running in ERM mode (centralized baseline).")
    log_file.write("Running in ERM mode (centralized baseline).\n")
    all_metrics = algorithms.run_erm_training(global_model, clients_train_datasets, config, device, log_file)
else:
    print("Running in Federated mode.")
    log_file.write("Running in Federated mode.\n")
    all_metrics = algorithms.run_federated_training(global_model, clients, config, device, log_file)

metrics.save_training_plots(all_metrics, output_dir, rounds_per_task,
                            num_tasks=len(clients_train_datasets[0]) if config['dataset'].get('holdout_last_task', True)
                                    else len(clients_train_datasets[0]))
print("\nTraining on all tasks completed.")
log_file.write("\nTraining on all tasks completed.\n")
log_file.close()

print("\n=== Starting Final Evaluation on Held-out Test Data ===")
server.evaluate_global()
