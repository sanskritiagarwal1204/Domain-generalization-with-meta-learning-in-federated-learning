import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    ndcg_score,
    precision_recall_curve
)

def evaluate_model(model, test_loader, device=torch.device("cpu"), use_adapter=False,threshold=0.25):
    """
    Evaluate a model on a test set.
    
    Args:
      model: The trained model.
      test_loader: DataLoader for test data.
      device: Torch device to run evaluation on.
      use_adapter: If True, use the adapter-based forward pass (zero-shot adaptation).
    
    Returns:
      accuracy (float): Overall accuracy.
      all_preds (list): Predicted labels.
      all_labels (list): Ground-truth labels.
      all_probs (list): Predicted probability (for the positive class).
    """
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for (cat_feats, num_feats), labels in test_loader:
            cat_feats = cat_feats.to(device)
            num_feats = num_feats.to(device)
            labels = labels.to(device)
            
            if use_adapter:
                outputs = model.adapt_forward(cat_feats, num_feats)
            else:
                outputs = model(cat_feats, num_feats, use_global_stats=False)
            
            # Get probabilities via softmax
            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:,1] > threshold).long()
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            # Assuming binary classification; get probability for class "1"
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, all_preds, all_labels, all_probs

def compute_classification_metrics(y_true, y_prob):
    """
    Compute classification metrics including accuracy, precision, recall, F1 score,
    ROC-AUC, average precision, and NDCG.
    
    Args:
      y_true (list/array): Ground-truth labels.
      y_pred (list/array): Predicted labels.
      y_prob (list/array): Predicted probabilities for the positive class.
      
    Returns:
      dict: Dictionary containing the computed metrics.
    """

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Generate new predictions using optimal threshold
    y_pred = (np.array(y_prob) >= optimal_threshold).astype(int)
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics['roc_auc'] = None
        
    try:
        metrics['average_precision'] = average_precision_score(y_true, y_prob)
    except Exception:
        metrics['average_precision'] = None
        
    try:
        # ndcg_score expects 2D arrays: here each is reshaped to (1, n_samples)
        y_true_2d = np.array(y_true).reshape(1, -1)
        y_prob_2d = np.array(y_prob).reshape(1, -1)
        metrics['ndcg'] = ndcg_score(y_true_2d, y_prob_2d)
    except Exception:
        metrics['ndcg'] = None
        
    return metrics

def compute_fairness_metrics(model, test_loader, group_func, device=torch.device("cpu"), use_adapter=False,threshold=0.25):
    """
    Compute per-group accuracies as a fairness metric.
    
    Args:
      model: The trained model.
      test_loader: DataLoader for test data.
      group_func: Function that takes (cat_feats, label) for one sample and returns a group key.
      device: Torch device.
      use_adapter: If True, uses the adapter-based forward pass.
    
    Returns:
      dict: Dictionary mapping each group to its accuracy.
    """
    model.eval()
    group_correct, group_total = {}, {}
    
    with torch.no_grad():
        for (cat_feats, num_feats), labels in test_loader:
            cat_feats = cat_feats.to(device)
            num_feats = num_feats.to(device)
            labels = labels.to(device)
            
            if use_adapter:
                outputs = model.adapt_forward(cat_feats, num_feats)
            else:
                outputs = model(cat_feats, num_feats, use_global_stats=False)
            
            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:,1] > threshold).long()
            for i in range(labels.size(0)):
                group = group_func(cat_feats[i].cpu().numpy(), labels[i].item())
                group_total[group] = group_total.get(group, 0) + 1
                if preds[i] == labels[i]:
                    group_correct[group] = group_correct.get(group, 0) + 1
                    
    group_accuracy = {g: group_correct.get(g, 0)/group_total[g] for g in group_total if group_total[g] > 0}
    return group_accuracy
  