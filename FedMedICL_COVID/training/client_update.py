import torch

def train_FedDG_local(model, data_loader, criterion, optimizer, device=torch.device("cpu"), amp=False):
    """
    Local training for one client for one round under FedDG algorithm.
    Performs mixed BN training (local + global stats) on each batch.
    Optionally uses mixed-precision training if amp=True.
    """
    model.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler() if amp else None
    for (cat_feats, num_feats), labels in data_loader:
        cat_feats, num_feats, labels = cat_feats.to(device), num_feats.to(device), labels.to(device)
        if amp:
            with torch.cuda.amp.autocast():
                outputs_local = model(cat_feats, num_feats, use_global_stats=False)
                loss_local = criterion(outputs_local, labels)
                outputs_global = model(cat_feats, num_feats, use_global_stats=True)
                loss_global = criterion(outputs_global, labels)
                loss = loss_local + loss_global
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs_local = model(cat_feats, num_feats, use_global_stats=False)
            loss_local = criterion(outputs_local, labels)
            outputs_global = model(cat_feats, num_feats, use_global_stats=True)
            loss_global = criterion(outputs_global, labels)
            loss = loss_local + loss_global
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(data_loader.dataset)

def train_adapter_local(model, data_loader, criterion, adapter_optimizer, device=torch.device("cpu")):
    """
    Local training for BN adapter on one client (model frozen, adapter unfrozen).
    """
    model.eval()  # freeze main model
    model.bn_local.eval()
    total_loss = 0.0
    for (cat_feats, num_feats), labels in data_loader:
        cat_feats, num_feats, labels = cat_feats.to(device), num_feats.to(device), labels.to(device)
        with torch.no_grad():
            hidden = model.extract_features(cat_feats, num_feats)
            inst_mean = hidden.mean(dim=0)
            inst_var = hidden.var(dim=0, unbiased=False)
            global_mean = model.bn_global.running_mean.detach()
            global_var = model.bn_global.running_var.detach()
        stats_diff = torch.cat([inst_mean - global_mean, inst_var - global_var], dim=0)
        alpha = model.bn_adapter(stats_diff)
        mu_mix = alpha * inst_mean + (1 - alpha) * global_mean
        var_mix = alpha * inst_var + (1 - alpha) * global_var
        mu_mix = mu_mix.unsqueeze(0)
        var_mix = var_mix.unsqueeze(0)
        hidden_norm = (hidden - mu_mix) / torch.sqrt(var_mix + 1e-5)
        hidden_norm = hidden_norm * model.bn_local.weight + model.bn_local.bias
        hidden_act = torch.relu(hidden_norm)
        outputs = model.classifier(hidden_act)
        loss = criterion(outputs, labels)
        adapter_optimizer.zero_grad()
        loss.backward()
        adapter_optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(data_loader.dataset)
