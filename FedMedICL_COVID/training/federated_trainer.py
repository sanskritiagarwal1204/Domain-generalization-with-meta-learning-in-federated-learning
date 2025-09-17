import copy
import torch
from training.client_update import train_FedDG_local, train_adapter_local
from tqdm.auto import tqdm

def federated_train_FedDG(global_model, clients_datasets, rounds=50, adapter_rounds=5, lr=1e-2, adapter_lr=5e-3, weight_decay=0.0, amp=False, device=torch.device("cpu"), log=None):
    """
    Federated training loop for FedDG algorithm.
    clients_datasets: dict of {client_id: DataLoader}.
    Processes clients sequentially with progress bars.
    """
    global_model.to(device)
    optimizer_state = None

    for r in range(rounds):
        if log:
            log.info(f"Round {r+1}/{rounds}")
        else:
            print(f"Round {r+1}/{rounds}")
        client_updates = []
        total_samples = 0
        for cid, data_loader in tqdm(clients_datasets.items(), desc=f"Round {r+1}/{rounds}"):
            local_model = copy.deepcopy(global_model)
            local_model.to(device)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            if optimizer_state:
                optimizer.load_state_dict(optimizer_state)
            criterion = torch.nn.CrossEntropyLoss()
            loss = train_FedDG_local(local_model, data_loader, criterion, optimizer, device, amp=amp)
            if log:
                log.info(f" Client {cid} local loss: {loss:.4f}")
            else:
                print(f" Client {cid} local loss: {loss:.4f}")
            num_samples = len(data_loader.dataset)
            client_updates.append((num_samples, local_model.state_dict()))
            total_samples += num_samples
        
        # Aggregate model parameters (FedAvg)
        new_state_dict = copy.deepcopy(global_model.state_dict())
        # Initialize aggregated weights for floating point types to zero.
        for key in new_state_dict.keys():
            if new_state_dict[key].dtype in [torch.float16, torch.float32, torch.float64]:
                new_state_dict[key] = torch.zeros_like(new_state_dict[key])
        # Weighted average for float tensors; copy non-float buffers directly.
        for num_samples, state_dict in client_updates:
            for key, param in state_dict.items():
                if new_state_dict[key].dtype in [torch.float16, torch.float32, torch.float64]:
                    new_state_dict[key] += (num_samples / total_samples) * param
                else:
                    new_state_dict[key] = param  # simply copy non-floating point buffers
        global_model.load_state_dict(new_state_dict)
        # Update global BN stats to match local statistics.
        global_model.bn_global.running_mean = global_model.bn_local.running_mean.clone().detach()
        global_model.bn_global.running_var  = global_model.bn_local.running_var.clone().detach()
        optimizer_state = optimizer.state_dict()

    if adapter_rounds > 0:
        if log:
            log.info("Starting adapter fine-tuning phase...")
        else:
            print("Starting adapter fine-tuning phase...")
        for ar in range(adapter_rounds):
            client_adapters = []
            total_samples = 0
            for cid, data_loader in tqdm(clients_datasets.items(), desc=f"Adapter Round {ar+1}/{adapter_rounds}"):
                local_model = copy.deepcopy(global_model)
                local_model.to(device)
                # Freeze all parameters except those in the BN adapter.
                for name, param in local_model.named_parameters():
                    if "bn_adapter" not in name:
                        param.requires_grad = False
                adapter_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, local_model.parameters()), lr=adapter_lr)
                criterion = torch.nn.CrossEntropyLoss()
                adapter_loss = train_adapter_local(local_model, data_loader, criterion, adapter_optimizer, device)
                if log:
                    log.info(f" Client {cid} adapter loss: {adapter_loss:.4f}")
                else:
                    print(f" Client {cid} adapter loss: {adapter_loss:.4f}")
                num_samples = len(data_loader.dataset)
                client_adapters.append((num_samples, {name: param for name, param in local_model.state_dict().items() if "bn_adapter" in name}))
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
            log.info("Adapter fine-tuning completed.")
        else:
            print("Adapter fine-tuning completed.")
    
    return global_model
