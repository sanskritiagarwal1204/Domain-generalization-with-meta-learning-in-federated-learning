# utils/saver.py

import os
import yaml
import json
import torch
import shutil
import matplotlib.pyplot as plt

def create_experiment_folder(base_dir, algorithm, rounds, extra_info=None):
    """
    Creates an experiment folder under base_dir using the algorithm name and rounds.
    For example, if algorithm is 'FedDG' and rounds is 150, the folder will be named 'feddg_150R'.
    An optional extra_info string can be appended.
    """
    experiment_name = f"{algorithm.lower()}_{rounds}R"
    if extra_info:
        experiment_name += "_" + extra_info
    experiment_folder = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_folder, exist_ok=True)
    return experiment_folder

def save_config(cfg, save_path):
    """
    Save the configuration dictionary as a YAML file.
    """
    with open(save_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

def save_metrics(metrics_dict, save_path):
    """
    Save metrics dictionary as a JSON file.
    """
    with open(save_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

def save_model(model, save_path):
    """
    Save the PyTorch model state dictionary.
    """
    torch.save(model.state_dict(), save_path)

def copy_logs(source_log_path, dest_folder):
    """
    Copy a log file from source_log_path into dest_folder.
    """
    shutil.copy(source_log_path, dest_folder)

def save_plot(figure, save_path):
    """
    Save a matplotlib figure to the specified path.
    
    Args:
        figure: a matplotlib.figure.Figure object.
        save_path: Path to save the plot image (e.g., PNG).
    """
    figure.savefig(save_path)
    plt.close(figure)
