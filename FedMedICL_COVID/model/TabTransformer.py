import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BNAdapter import BNAdapter

class TabTransformer(nn.Module):
    """
    TabTransformer model for tabular data with categorical and numerical features.
    Incorporates dual BatchNorm (local & global) for federated adaptation.
    """
    def __init__(self, num_categories:list, num_numeric:int, emb_dim:int=64, 
                 transformer_heads:int=8, transformer_layers:int=4, hidden_dim:int=512, 
                 num_classes:int=2, dropout:float=0.0):
        super(TabTransformer, self).__init__()
        self.num_cat = len(num_categories)
        self.num_numeric = num_numeric
        # Embedding layers for each categorical feature
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(num_categories[i], emb_dim) for i in range(self.num_cat)
        ])
        # Transformer to process categorical feature embeddings
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=transformer_heads, 
                                                   dim_feedforward=emb_dim*2, dropout=dropout, 
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        # BatchNorm for numeric features (optional)
        if self.num_numeric > 0:
            self.numeric_bn = nn.BatchNorm1d(self.num_numeric)
        else:
            self.numeric_bn = None
        # Fully connected layers: first combine transformer output + numeric features
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.num_cat * emb_dim + num_numeric, hidden_dim)
        # Dual BatchNorm layers for hidden representations
        self.bn_local = nn.BatchNorm1d(hidden_dim)   # local BN (learnable affine)
        self.bn_global = nn.BatchNorm1d(hidden_dim, affine=False)  # global BN (no affine, just stats)
        # Adapter for BN layer
        self.bn_adapter = BNAdapter(hidden_dim)
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # Initialize global BN stats same as local
        self._init_global_bn()

    def _init_global_bn(self):
        self.bn_global.running_mean = self.bn_local.running_mean.clone().detach()
        self.bn_global.running_var  = self.bn_local.running_var.clone().detach()
        self.bn_global.num_batches_tracked = self.bn_local.num_batches_tracked.clone().detach()

    def forward(self, cat_feats, num_feats=None, use_global_stats=False):
        batch_size = cat_feats.size(0)
        embs = [] 
        for j, emb_layer in enumerate(self.cat_embeds):
            fea = cat_feats[:, j]
            embs.append(emb_layer(fea))
        if len(embs) > 0:
            cat_emb_matrix = torch.stack(embs, dim=1)
            cat_context = self.transformer(cat_emb_matrix)
            cat_flat = cat_context.reshape(batch_size, -1)
        else:
            cat_flat = torch.zeros((batch_size, 0), device=cat_feats.device)
        if num_feats is not None and self.num_numeric > 0:
            if self.numeric_bn is not None:
                num_norm = self.numeric_bn(num_feats)
            else:
                num_norm = num_feats
        else:
            num_norm = torch.zeros((batch_size, 0), device=cat_feats.device)
        features = torch.cat([cat_flat, num_norm], dim=1)
        hidden = self.fc1(features)
        if use_global_stats:
            hidden_norm = F.batch_norm(hidden, self.bn_global.running_mean, self.bn_global.running_var, 
                                       weight=self.bn_local.weight, bias=self.bn_local.bias, training=False)
        else:
            hidden_norm = self.bn_local(hidden)
        hidden_act = F.relu(hidden_norm)
        logits = self.classifier(hidden_act)
        return logits

    def adapt_forward(self, cat_feats, num_feats=None):
        batch_size = cat_feats.size(0)
        embs = [emb_layer(cat_feats[:, j]) for j, emb_layer in enumerate(self.cat_embeds)]
        cat_flat = torch.stack(embs, dim=1).reshape(batch_size, -1) if len(embs)>0 else torch.zeros((batch_size,0), device=cat_feats.device)
        if num_feats is not None and self.num_numeric > 0:
            num_norm = self.numeric_bn(num_feats) if self.numeric_bn else num_feats
        else:
            num_norm = torch.zeros((batch_size, 0), device=cat_feats.device)
        features = torch.cat([cat_flat, num_norm], dim=1)
        hidden = self.fc1(features)
        inst_mean = hidden.mean(dim=0)
        inst_var = hidden.var(dim=0, unbiased=False)
        global_mean = self.bn_global.running_mean.detach()
        global_var = self.bn_global.running_var.detach()
        alpha = self.bn_adapter(torch.cat([inst_mean - global_mean, inst_var - global_var], dim=0))
        mu_mix = alpha * inst_mean + (1 - alpha) * global_mean
        var_mix = alpha * inst_var + (1 - alpha) * global_var
        mu_mix = mu_mix.unsqueeze(0).expand(batch_size, -1)
        var_mix = var_mix.unsqueeze(0).expand(batch_size, -1)
        hidden_adapt = (hidden - mu_mix) / torch.sqrt(var_mix + 1e-5)
        hidden_adapt = hidden_adapt * self.bn_local.weight + self.bn_local.bias
        hidden_act = F.relu(hidden_adapt)
        logits = self.classifier(hidden_act)
        return logits

    def extract_features(self, cat_feats, num_feats=None):
        embs = [emb_layer(cat_feats[:, j]) for j, emb_layer in enumerate(self.cat_embeds)]
        batch_size = cat_feats.size(0)
        cat_flat = torch.stack(embs, dim=1).reshape(batch_size, -1) if len(embs)>0 else torch.zeros((batch_size,0), device=cat_feats.device)
        if num_feats is not None and self.num_numeric > 0:
            num_norm = self.numeric_bn(num_feats) if self.numeric_bn else num_feats
        else:
            num_norm = torch.zeros((batch_size, 0), device=cat_feats.device)
        features = torch.cat([cat_flat, num_norm], dim=1)
        hidden = self.fc1(features)
        return hidden
