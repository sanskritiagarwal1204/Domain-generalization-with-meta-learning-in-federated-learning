import torch
import torch.nn as nn

class BNAdapter(nn.Module):
    """
    Adapter network that predicts interpolation factor alpha for BN statistics.
    Input: concatenated (inst_mean - global_mean) and (inst_var - global_var) for a layer (both are vectors of length C).
    Output: alpha vector of length C in [0,1] for that layer.
    """
    def __init__(self, num_features:int):
        super(BNAdapter, self).__init__()
        self.num_features = num_features
        # Small MLP: input 2*num_features, output num_features
        hidden = max(4, num_features // 2)  # heuristic: small hidden size
        self.net = nn.Sequential(
            nn.Linear(2 * num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_features),
            nn.Sigmoid()
        )
    def forward(self, stats_diff):
        # stats_diff expected shape: (2*num_features,) or (batch, 2*num_features) if batch dimension
        return self.net(stats_diff)
