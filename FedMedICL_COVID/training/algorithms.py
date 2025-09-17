# training/algorithms.py

import copy
import torch
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import recall_score, precision_score, f1_score

class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss.
    Args:
        gamma (float): Focusing parameter for modulating factor (1-pt)^gamma.
        alpha (Tensor, optional): Weighting factor in range [0, 1] for each class.
        reduction (str): 'mean', 'sum', or 'none'.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # can be a tensor for per-class weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute softmax over the logits
        logpt = F.log_softmax(inputs, dim=1)
        pt = torch.exp(logpt)
        
        # Gather log probabilities for the target class
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # If alpha is provided, use it to weight the loss
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets)
            logpt = logpt * at
        
        loss = -1 * (1 - pt) ** self.gamma * logpt
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def federated_train_FedAvg(global_model, clients_datasets, rounds=50, lr=1e-2, weight_decay=0.0, amp=False, device=torch.device("cpu"), log=None):
    """
    Simple Federated Averaging (FedAvg) training loop.
    Clients train using a standard forward pass and loss (without domain-specific adaptations),
    and the global model is updated via weighted averaging.
    """
    global_model.to(device)
    optimizer_state = None

    for r in range(rounds):
        if log:
            log.info(f"Round {r+1}/{rounds} (FedAvg)")
        else:
            print(f"Round {r+1}/{rounds} (FedAvg)")
        client_updates = []
        total_samples = 0

        for cid, data_loader in tqdm(clients_datasets.items(), desc=f"Round {r+1}/{rounds} (FedAvg)"):
            local_model = copy.deepcopy(global_model)
            local_model.to(device)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            criterion = torch.nn.CrossEntropyLoss()
            total_loss = 0.0
            local_model.train()
            scaler = torch.cuda.amp.GradScaler() if amp else None

            for (cat_feats, num_feats), labels in data_loader:
                cat_feats, num_feats, labels = cat_feats.to(device), num_feats.to(device), labels.to(device)
                if amp:
                    with torch.cuda.amp.autocast():
                        outputs = local_model(cat_feats, num_feats, use_global_stats=False)
                        loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = local_model(cat_feats, num_feats, use_global_stats=False)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * labels.size(0)
            avg_loss = total_loss / len(data_loader.dataset)
            if log:
                log.info(f" Client {cid} local loss (FedAvg): {avg_loss:.4f}")
            else:
                print(f" Client {cid} local loss (FedAvg): {avg_loss:.4f}")
            num_samples = len(data_loader.dataset)
            client_updates.append((num_samples, local_model.state_dict()))
            total_samples += num_samples

        # FedAvg aggregation: weighted average of client parameters.
        new_state_dict = copy.deepcopy(global_model.state_dict())
        for key in new_state_dict.keys():
            if new_state_dict[key].dtype in [torch.float16, torch.float32, torch.float64]:
                new_state_dict[key] = torch.zeros_like(new_state_dict[key])
        for num_samples, state_dict in client_updates:
            for key, param in state_dict.items():
                if new_state_dict[key].dtype in [torch.float16, torch.float32, torch.float64]:
                    new_state_dict[key] += (num_samples / total_samples) * param
                else:
                    new_state_dict[key] = param
        global_model.load_state_dict(new_state_dict)
        optimizer_state = optimizer.state_dict()
    return global_model


def federated_train_FedDG(global_model, clients_datasets, rounds=50, adapter_rounds=5, lr=1e-2, adapter_lr=5e-3, weight_decay=0.0, amp=False, device=torch.device("cpu"), log=None):
    """
    Federated training loop for FedDG algorithm.
    Uses dual batch normalization (local and global) along with BN adapter fine-tuning.
    """
    global_model.to(device)
    optimizer_state = None
    for r in range(rounds):
        if log:
            log.info(f"Round {r+1}/{rounds} (FedDG)")
        else:
            print(f"Round {r+1}/{rounds} (FedDG)")
        client_updates = []
        total_samples = 0
        for cid, data_loader in tqdm(clients_datasets.items(), desc=f"Round {r+1}/{rounds} (FedDG)"):
            local_model = copy.deepcopy(global_model)
            local_model.to(device)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            criterion = torch.nn.CrossEntropyLoss()
            total_loss = 0.0
            local_model.train()
            scaler = torch.cuda.amp.GradScaler() if amp else None
            for (cat_feats, num_feats), labels in data_loader:
                cat_feats, num_feats, labels = cat_feats.to(device), num_feats.to(device), labels.to(device)
                if amp:
                    with torch.cuda.amp.autocast():
                        outputs_local = local_model(cat_feats, num_feats, use_global_stats=False)
                        loss_local = criterion(outputs_local, labels)
                        outputs_global = local_model(cat_feats, num_feats, use_global_stats=True)
                        loss_global = criterion(outputs_global, labels)
                        loss = loss_local + loss_global
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs_local = local_model(cat_feats, num_feats, use_global_stats=False)
                    loss_local = criterion(outputs_local, labels)
                    outputs_global = local_model(cat_feats, num_feats, use_global_stats=True)
                    loss_global = criterion(outputs_global, labels)
                    loss = loss_local + loss_global
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * labels.size(0)
            avg_loss = total_loss / len(data_loader.dataset)
            if log:
                log.info(f" Client {cid} local loss (FedDG): {avg_loss:.4f}")
            else:
                print(f" Client {cid} local loss (FedDG): {avg_loss:.4f}")
            num_samples = len(data_loader.dataset)
            client_updates.append((num_samples, local_model.state_dict()))
            total_samples += num_samples

        # Aggregate parameters using FedAvg (only averaging floating point values)
        new_state_dict = copy.deepcopy(global_model.state_dict())
        for key in new_state_dict.keys():
            if new_state_dict[key].dtype in [torch.float16, torch.float32, torch.float64]:
                new_state_dict[key] = torch.zeros_like(new_state_dict[key])
        for num_samples, state_dict in client_updates:
            for key, param in state_dict.items():
                if new_state_dict[key].dtype in [torch.float16, torch.float32, torch.float64]:
                    new_state_dict[key] += (num_samples / total_samples) * param
                else:
                    new_state_dict[key] = param
        global_model.load_state_dict(new_state_dict)
        # Update BN global statistics from local BN
        global_model.bn_global.running_mean = global_model.bn_local.running_mean.clone().detach()
        global_model.bn_global.running_var  = global_model.bn_local.running_var.clone().detach()
        optimizer_state = optimizer.state_dict()

    # Adapter fine-tuning phase
    if adapter_rounds > 0:
        if log:
            log.info("Starting adapter fine-tuning phase (FedDG)...")
        else:
            print("Starting adapter fine-tuning phase (FedDG)...")
        for ar in range(adapter_rounds):
            client_adapters = []
            total_samples = 0
            for cid, data_loader in tqdm(clients_datasets.items(), desc=f"Adapter Round {ar+1}/{adapter_rounds} (FedDG)"):
                local_model = copy.deepcopy(global_model)
                local_model.to(device)
                # Freeze non-adapter parameters
                for name, param in local_model.named_parameters():
                    if "bn_adapter" not in name:
                        param.requires_grad = False
                adapter_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, local_model.parameters()), lr=adapter_lr)
                criterion = torch.nn.CrossEntropyLoss()
                total_loss = 0.0
                local_model.train()
                for (cat_feats, num_feats), labels in data_loader:
                    cat_feats, num_feats, labels = cat_feats.to(device), num_feats.to(device), labels.to(device)
                    outputs = local_model.adapt_forward(cat_feats, num_feats)
                    loss = criterion(outputs, labels)
                    adapter_optimizer.zero_grad()
                    loss.backward()
                    adapter_optimizer.step()
                    total_loss += loss.item() * labels.size(0)
                avg_loss = total_loss / len(data_loader.dataset)
                if log:
                    log.info(f" Client {cid} adapter loss (FedDG): {avg_loss:.4f}")
                else:
                    print(f" Client {cid} adapter loss (FedDG): {avg_loss:.4f}")
                num_samples = len(data_loader.dataset)
                adapter_params = {name: param for name, param in local_model.state_dict().items() if "bn_adapter" in name}
                client_adapters.append((num_samples, adapter_params))
                total_samples += num_samples
            new_adapter_params = {name: torch.zeros_like(param) for name, param in global_model.state_dict().items() if "bn_adapter" in name}
            for num_samples, adapter_params in client_adapters:
                for name, param in adapter_params.items():
                    new_adapter_params[name] += (num_samples / total_samples) * param
            current_state = global_model.state_dict()
            for name, param in new_adapter_params.items():
                current_state[name] = param
            global_model.load_state_dict(current_state)
        if log:
            log.info("Adapter fine-tuning completed (FedDG).")
        else:
            print("Adapter fine-tuning completed (FedDG).")
    return global_model

def federated_train_FedCB(global_model, clients_datasets, rounds=50, lr=1e-2, weight_decay=0.0, amp=False, device=torch.device("cpu"), log=None):
    """
    Federated training loop for FedCB (Federated Class-Balancing).
    Each client calculates class-balanced weights for the CrossEntropy loss.
    """
    global_model.to(device)
    optimizer_state = None
    for r in range(rounds):
        if log:
            log.info(f"Round {r+1}/{rounds} (FedCB)")
        else:
            print(f"Round {r+1}/{rounds} (FedCB)")
        client_updates = []
        total_samples = 0
        for cid, data_loader in tqdm(clients_datasets.items(), desc=f"Round {r+1}/{rounds} (FedCB)"):
            # Compute label counts and derive class-balanced weights.
            labels_all = data_loader.dataset.y  # assumed to be a NumPy array of labels
            counts = np.bincount(labels_all.astype(int))
            counts = np.where(counts == 0, 1, counts)  # avoid division by zero
            class_weights = 1.0 / counts
            class_weights = class_weights / np.mean(class_weights)
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
            
            local_model = copy.deepcopy(global_model)
            local_model.to(device)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            # Use a reweighted CrossEntropyLoss
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
            total_loss = 0.0
            local_model.train()
            scaler = torch.cuda.amp.GradScaler() if amp else None
            for (cat_feats, num_feats), labels in data_loader:
                cat_feats, num_feats, labels = cat_feats.to(device), num_feats.to(device), labels.to(device)
                if amp:
                    with torch.cuda.amp.autocast():
                        outputs = local_model(cat_feats, num_feats, use_global_stats=False)
                        loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = local_model(cat_feats, num_feats, use_global_stats=False)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * labels.size(0)
            avg_loss = total_loss / len(data_loader.dataset)
            if log:
                log.info(f" Client {cid} local loss (FedCB): {avg_loss:.4f}")
            else:
                print(f" Client {cid} local loss (FedCB): {avg_loss:.4f}")
            num_samples = len(data_loader.dataset)
            client_updates.append((num_samples, local_model.state_dict()))
            total_samples += num_samples
        
        # Aggregate clients with FedAvg
        new_state_dict = copy.deepcopy(global_model.state_dict())
        for key in new_state_dict.keys():
            if new_state_dict[key].dtype in [torch.float16, torch.float32, torch.float64]:
                new_state_dict[key] = torch.zeros_like(new_state_dict[key])
        for num_samples, state_dict in client_updates:
            for key, param in state_dict.items():
                if new_state_dict[key].dtype in [torch.float16, torch.float32, torch.float64]:
                    new_state_dict[key] += (num_samples / total_samples) * param
                else:
                    new_state_dict[key] = param
        global_model.load_state_dict(new_state_dict)
        optimizer_state = optimizer.state_dict()
    return global_model

def train_erm(global_model, merged_loader, epochs=10, lr=0.01, weight_decay=0.0, amp=False, device=torch.device("cpu"), log=None):
    """
    Centralized training for ERM (Empirical Risk Minimization) baseline.
    Trains the model on merged data for a given number of epochs.
    """
    global_model.to(device)
    optimizer = torch.optim.Adam(global_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if amp else None
    for epoch in range(epochs):
        global_model.train()
        total_loss = 0.0
        for (cat_feats, num_feats), labels in merged_loader:
            cat_feats, num_feats, labels = cat_feats.to(device), num_feats.to(device), labels.to(device)
            optimizer.zero_grad()
            if amp:
                with torch.cuda.amp.autocast():
                    outputs = global_model(cat_feats, num_feats, use_global_stats=False)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = global_model(cat_feats, num_feats, use_global_stats=False)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
        epoch_loss = total_loss / len(merged_loader.dataset)
        if log:
            log.info(f"ERM Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        else:
            print(f"ERM Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    return global_model

def federated_train_FedDGHybrid(
    global_model, 
    clients_datasets,
    rounds=50,
    adapter_rounds=5,
    lr=1e-2,
    adapter_lr=5e-3,
    weight_decay=0.0,
    gamma=2.0,
    amp=False,
    device=torch.device("cpu"),
    log=None
):
    """
    FedDG + Class Balancing + Focal Loss
    1) Dual BN usage (local vs. global) in forward pass.
    2) Class-balanced weights used to define alpha in a FocalLoss.
    3) After main training, do BN adapter fine-tuning like FedDG.
    """
    global_model.to(device)
    optimizer_state = None

    # -------------------- Main Federated Rounds --------------------
    for r in range(rounds):
        # Log or print
        client_updates = []
        total_samples = 0
        for cid, data_loader in clients_datasets.items():
            # 1) Compute per-client label frequency -> class-balanced alpha
            labels_all = data_loader.dataset.y
            counts = np.bincount(labels_all.astype(int))
            counts = np.where(counts == 0, 1, counts)
            class_weights = 1.0 / counts
            class_weights = class_weights / np.mean(class_weights)
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

            # 2) Copy global model -> local model
            local_model = copy.deepcopy(global_model).to(device)
            # 3) Build an optimizer, possibly reusing momentum
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)

            # 4) Build a FocalLoss with alpha=class_weights_tensor
            focal_loss_fn = FocalLoss(gamma=gamma, alpha=class_weights_tensor, reduction='mean')

            # 5) For each batch, do forward pass with local BN, then also with global BN (like FedDG),
            #    sum the losses, update.
            local_model.train()
            total_loss = 0.0
            scaler = torch.cuda.amp.GradScaler() if amp else None
            for (cat_feats, num_feats), labels in data_loader:
                cat_feats = cat_feats.to(device)
                num_feats = num_feats.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                if amp:
                    with torch.cuda.amp.autocast():
                        outputs_local = local_model(cat_feats, num_feats, use_global_stats=False)
                        loss_local = focal_loss_fn(outputs_local, labels)
                        outputs_global = local_model(cat_feats, num_feats, use_global_stats=True)
                        loss_global = focal_loss_fn(outputs_global, labels)
                        loss = loss_local + loss_global
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs_local = local_model(cat_feats, num_feats, use_global_stats=False)
                    loss_local = focal_loss_fn(outputs_local, labels)
                    outputs_global = local_model(cat_feats, num_feats, use_global_stats=True)
                    loss_global = focal_loss_fn(outputs_global, labels)
                    loss = loss_local + loss_global
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item() * labels.size(0)
            
            avg_loss = total_loss / len(data_loader.dataset)
            # log it
            num_samples = len(data_loader.dataset)
            client_updates.append((num_samples, local_model.state_dict()))
            total_samples += num_samples
        
        # 6) Aggregate with FedAvg
        new_state_dict = copy.deepcopy(global_model.state_dict())
        for k in new_state_dict.keys():
            if new_state_dict[k].dtype in [torch.float16, torch.float32, torch.float64]:
                new_state_dict[k] = torch.zeros_like(new_state_dict[k])
        for num_samples, state_dict in client_updates:
            for k, param in state_dict.items():
                if new_state_dict[k].dtype in [torch.float16, torch.float32, torch.float64]:
                    new_state_dict[k] += (num_samples / total_samples) * param
                else:
                    new_state_dict[k] = param
        
        global_model.load_state_dict(new_state_dict)
        # Also update BN global stats from BN local if your model does that
        global_model.bn_global.running_mean = global_model.bn_local.running_mean.clone().detach()
        global_model.bn_global.running_var  = global_model.bn_local.running_var.clone().detach()

        # save momentum
        optimizer_state = optimizer.state_dict()

    # -------------------- BN Adapter Fine-Tuning --------------------
    if adapter_rounds > 0:
        if log:
            log.info("Starting adapter fine-tuning phase (FedDG)...")
        else:
            print("Starting adapter fine-tuning phase (FedDG)...")
        for ar in range(adapter_rounds):
            client_adapters = []
            total_samples = 0
            for cid, data_loader in tqdm(clients_datasets.items(), desc=f"Adapter Round {ar+1}/{adapter_rounds} (FedDG)"):
                local_model = copy.deepcopy(global_model)
                local_model.to(device)
                # Freeze non-adapter parameters
                for name, param in local_model.named_parameters():
                    if "bn_adapter" not in name:
                        param.requires_grad = False
                adapter_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, local_model.parameters()), lr=adapter_lr)
                criterion = torch.nn.CrossEntropyLoss()
                total_loss = 0.0
                local_model.train()
                for (cat_feats, num_feats), labels in data_loader:
                    cat_feats, num_feats, labels = cat_feats.to(device), num_feats.to(device), labels.to(device)
                    outputs = local_model.adapt_forward(cat_feats, num_feats)
                    loss = criterion(outputs, labels)
                    adapter_optimizer.zero_grad()
                    loss.backward()
                    adapter_optimizer.step()
                    total_loss += loss.item() * labels.size(0)
                avg_loss = total_loss / len(data_loader.dataset)
                if log:
                    log.info(f" Client {cid} adapter loss (FedDG): {avg_loss:.4f}")
                else:
                    print(f" Client {cid} adapter loss (FedDG): {avg_loss:.4f}")
                num_samples = len(data_loader.dataset)
                adapter_params = {name: param for name, param in local_model.state_dict().items() if "bn_adapter" in name}
                client_adapters.append((num_samples, adapter_params))
                total_samples += num_samples
            new_adapter_params = {name: torch.zeros_like(param) for name, param in global_model.state_dict().items() if "bn_adapter" in name}
            for num_samples, adapter_params in client_adapters:
                for name, param in adapter_params.items():
                    new_adapter_params[name] += (num_samples / total_samples) * param
            current_state = global_model.state_dict()
            for name, param in new_adapter_params.items():
                current_state[name] = param
            global_model.load_state_dict(current_state)
        if log:
            log.info("Adapter fine-tuning completed (FedDG).")
        else:
            print("Adapter fine-tuning completed (FedDG).")
    return global_model


def federated_train_FedAIM(
    global_model,
    clients_datasets,
    rounds=50,
    adapter_rounds=5,
    lr=1e-2,
    adapter_lr=5e-3,
    weight_decay=0.0,
    amp=False,
    lambda_consist=0.5,  # weight for the consistency loss term
    beta_kd=0.5,         # weight for the knowledge distillation loss term
    meta_eta=0.1,        # meta adaptation learning rate for positive class weight
    target_recall=0.7,   # target recall for the positive class (to guide adaptation)
    device=torch.device("cpu"),
    log=None
):
    """
    FedAIM: Federated Adaptive Imbalance Mitigation with Meta-Learned Loss Adaptation.

    This algorithm improves F1-score on imbalanced tabular data by:
      1. Dynamically reweighting samples based on difficulty (using teacher predictions).
      2. Enforcing consistency between local (client) and global BN outputs.
      3. Applying knowledge distillation from a global teacher model.
      4. Meta-learning the loss weighting: using aggregated global evaluation metrics to update
         a global meta-class weight for the positive (minority) class.
      5. Aggregating client updates via FedAvg and then performing BNAdapter fine-tuning.

    Args:
      global_model: the shared global model (TabTransformer) with dual BN and BNAdapter.
      clients_datasets: dict mapping client IDs to DataLoader objects.
      rounds (int): number of federated communication rounds.
      adapter_rounds (int): number of adapter fine-tuning rounds.
      lr (float): learning rate for main training.
      adapter_lr (float): learning rate for adapter fine-tuning.
      weight_decay (float): weight decay for optimizer.
      amp (bool): whether to use mixed precision.
      lambda_consist (float): weight for the consistency loss.
      beta_kd (float): weight for the knowledge distillation loss.
      meta_eta (float): learning rate for meta adaptation of the positive class weight.
      target_recall (float): desired recall for the minority (positive) class.
      device: torch device.
      log: logger object (optional).

    Returns:
      global_model: the updated global model after training.
    """
    global_model.to(device)
    optimizer_state = None

    # Initialize meta-class weights as a tensor [w_negative, w_positive]
    # We start with 1.0 for both classes.
    meta_class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)

    for r in range(rounds):
        if log:
            log.info(f"Round {r+1}/{rounds} (FedAIM)")
        else:
            print(f"Round {r+1}/{rounds} (FedAIM)")
        client_updates = []
        total_samples = 0

        # For each client: perform local training with adaptive loss
        for cid, data_loader in tqdm(clients_datasets.items(), desc=f"Round {r+1}/{rounds} (FedAIM)"):
            # -- Create teacher model as a copy of current global model --
            teacher_model = copy.deepcopy(global_model)
            teacher_model.to(device)
            teacher_model.eval()  # teacher uses global BN for predictions

            # -- Create a local model copy --
            local_model = copy.deepcopy(global_model)
            local_model.to(device)

            # Set up optimizer (SGD with momentum)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)

            # Use per-sample CrossEntropyLoss (no reduction) to compute difficulty scores
            criterion = nn.CrossEntropyLoss(reduction='none')

            local_model.train()
            total_loss = 0.0
            scaler = torch.cuda.amp.GradScaler() if amp else None

            # Iterate over batches
            for (cat_feats, num_feats), labels in data_loader:
                cat_feats = cat_feats.to(device)
                num_feats = num_feats.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # -- Knowledge Distillation: get teacher predictions --
                with torch.no_grad():
                    teacher_outputs = teacher_model(cat_feats, num_feats, use_global_stats=True)
                    teacher_probs = torch.softmax(teacher_outputs, dim=1)
                    # Get teacher probability for the true label for each sample
                    p_true = teacher_probs.gather(1, labels.unsqueeze(1)).squeeze(1)

                # -- Compute dynamic sample weights using meta_class_weights --
                # Use the meta-class weight for each sample (meta_class_weights is [w_neg, w_pos])
                sample_cw = meta_class_weights.gather(0, labels)
                # Weight each sample by (1 - teacher probability) so that hard examples get higher weight
                sample_weights = sample_cw * (1 - p_true)  # shape: (batch_size,)

                if amp:
                    with torch.cuda.amp.autocast():
                        # Forward pass with local BN (client-specific)
                        outputs_local = local_model(cat_feats, num_feats, use_global_stats=False)
                        # Forward pass with global BN (using global statistics)
                        outputs_global = local_model(cat_feats, num_feats, use_global_stats=True)

                        # Compute per-sample loss (classification)
                        loss_local = criterion(outputs_local, labels)
                        loss_global = criterion(outputs_global, labels)
                        # Weight each loss per sample and average
                        loss_local = torch.mean(sample_weights * loss_local)
                        loss_global = torch.mean(sample_weights * loss_global)

                        # Consistency loss: KL divergence between local and global predictions
                        cons_loss = F.kl_div(torch.log_softmax(outputs_local, dim=1),
                                               torch.softmax(outputs_global, dim=1),
                                               reduction='batchmean')
                        # KD loss: distill teacher's soft labels from global pathway
                        kd_loss = F.kl_div(torch.log_softmax(outputs_global, dim=1),
                                             teacher_probs,
                                             reduction='batchmean')

                        loss = loss_local + loss_global + lambda_consist * cons_loss + beta_kd * kd_loss

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs_local = local_model(cat_feats, num_feats, use_global_stats=False)
                    outputs_global = local_model(cat_feats, num_feats, use_global_stats=True)
                    loss_local = criterion(outputs_local, labels)
                    loss_global = criterion(outputs_global, labels)
                    loss_local = torch.mean(sample_weights * loss_local)
                    loss_global = torch.mean(sample_weights * loss_global)
                    cons_loss = F.kl_div(torch.log_softmax(outputs_local, dim=1),
                                         torch.softmax(outputs_global, dim=1),
                                         reduction='batchmean')
                    kd_loss = F.kl_div(torch.log_softmax(outputs_global, dim=1),
                                       teacher_probs,
                                       reduction='batchmean')
                    loss = loss_local + loss_global + lambda_consist * cons_loss + beta_kd * kd_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * labels.size(0)

            avg_loss = total_loss / len(data_loader.dataset)
            if log:
                log.info(f" Client {cid} local loss (FedAIM): {avg_loss:.4f}")
            else:
                print(f" Client {cid} local loss (FedAIM): {avg_loss:.4f}")

            num_samples = len(data_loader.dataset)
            client_updates.append((num_samples, local_model.state_dict()))
            total_samples += num_samples

        # ----- Server Aggregation using FedAvg -----
        new_state_dict = copy.deepcopy(global_model.state_dict())
        for key in new_state_dict.keys():
            if new_state_dict[key].dtype in [torch.float16, torch.float32, torch.float64]:
                new_state_dict[key] = torch.zeros_like(new_state_dict[key])
        for num_samples, state_dict in client_updates:
            for key, param in state_dict.items():
                if new_state_dict[key].dtype in [torch.float16, torch.float32, torch.float64]:
                    new_state_dict[key] += (num_samples / total_samples) * param
                else:
                    new_state_dict[key] = param
        global_model.load_state_dict(new_state_dict)

        # Update BN global statistics with current local BN stats
        global_model.bn_global.running_mean = global_model.bn_local.running_mean.clone().detach()
        global_model.bn_global.running_var  = global_model.bn_local.running_var.clone().detach()

        optimizer_state = optimizer.state_dict()
        if log:
            log.info(f"Completed round {r+1}/{rounds} for FedAIM.")
        else:
            print(f"Completed round {r+1}/{rounds} for FedAIM.")

        # ----- Meta-Learned Loss Adaptation Step -----
        # Aggregate predictions over all clients to compute global metrics, then update meta_class_weights
        all_preds = []
        all_labels = []
        for cid, data_loader in clients_datasets.items():
            for (cat_feats, num_feats), labels in data_loader:
                cat_feats = cat_feats.to(device)
                num_feats = num_feats.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    outputs = global_model(cat_feats, num_feats, use_global_stats=False)
                    preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        # Compute global recall and precision for the positive class (assume positive class label == 1)
        global_recall = recall_score(all_labels, all_preds, pos_label=1)
        global_precision = precision_score(all_labels, all_preds, pos_label=1)
        global_f1 = f1_score(all_labels, all_preds, pos_label=1)
        if log:
            log.info(f"Meta Evaluation after round {r+1}: Recall (pos) = {global_recall:.4f}, Precision (pos) = {global_precision:.4f}, F1 (pos) = {global_f1:.4f}")
        else:
            print(f"Meta Evaluation after round {r+1}: Recall (pos) = {global_recall:.4f}, Precision (pos) = {global_precision:.4f}, F1 (pos) = {global_f1:.4f}")

        # Update the meta weight for the positive class if recall is below the target.
        # A simple heuristic: if recall < target, increase positive weight; if above, decrease slightly.
        if global_recall < target_recall:
            meta_class_weights[1] *= (1 + meta_eta * (target_recall - global_recall))
        else:
            meta_class_weights[1] *= max(0.5, (1 - meta_eta * (global_recall - target_recall)))
        if log:
            log.info(f"Updated meta_class_weights: {meta_class_weights.cpu().numpy()}")
        else:
            print(f"Updated meta_class_weights: {meta_class_weights.cpu().numpy()}")

    # ----- BN Adapter Fine-Tuning Phase (as before) -----
    if adapter_rounds > 0:
        if log:
            log.info("Starting adapter fine-tuning phase (FedAIM)...")
        else:
            print("Starting adapter fine-tuning phase (FedAIM)...")
        for ar in range(adapter_rounds):
            client_adapters = []
            total_samples = 0
            for cid, data_loader in tqdm(clients_datasets.items(), desc=f"Adapter Round {ar+1}/{adapter_rounds} (FedAIM)"):
                local_model = copy.deepcopy(global_model)
                local_model.to(device)
                # Freeze parameters that are not part of the BNAdapter.
                for name, param in local_model.named_parameters():
                    if "bn_adapter" not in name:
                        param.requires_grad = False
                adapter_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, local_model.parameters()), lr=adapter_lr)
                criterion = nn.CrossEntropyLoss()
                total_loss = 0.0
                local_model.train()
                for (cat_feats, num_feats), labels in data_loader:
                    cat_feats, num_feats, labels = cat_feats.to(device), num_feats.to(device), labels.to(device)
                    outputs = local_model.adapt_forward(cat_feats, num_feats)
                    loss = criterion(outputs, labels)
                    adapter_optimizer.zero_grad()
                    loss.backward()
                    adapter_optimizer.step()
                    total_loss += loss.item() * labels.size(0)
                avg_loss = total_loss / len(data_loader.dataset)
                if log:
                    log.info(f" Client {cid} adapter loss (FedAIM): {avg_loss:.4f}")
                else:
                    print(f" Client {cid} adapter loss (FedAIM): {avg_loss:.4f}")
                num_samples = len(data_loader.dataset)
                adapter_params = {name: param for name, param in local_model.state_dict().items() if "bn_adapter" in name}
                client_adapters.append((num_samples, adapter_params))
                total_samples += num_samples
            new_adapter_params = {name: torch.zeros_like(param) for name, param in global_model.state_dict().items() if "bn_adapter" in name}
            for num_samples, adapter_params in client_adapters:
                for name, param in adapter_params.items():
                    new_adapter_params[name] += (num_samples / total_samples) * param
            current_state = global_model.state_dict()
            for name, param in new_adapter_params.items():
                current_state[name] = param
            global_model.load_state_dict(current_state)
        if log:
            log.info("Adapter fine-tuning completed (FedAIM).")
        else:
            print("Adapter fine-tuning completed (FedAIM).")
    return global_model

