import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear, Dropout

class GAT(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_classes,
        num_heads=2,
        dropout_rate=0.015,
        global_feature_dim=0,
    ):
        super(GAT, self).__init__()
        self.global_feature_dim = int(global_feature_dim)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * num_heads + self.global_feature_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(32, num_classes)
        )
        self.dropout = Dropout(dropout_rate)

    def forward(self, data, return_pooled=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.gat1(x, edge_index))
        x = self.dropout(x)

        x = F.relu(self.gat2(x, edge_index))

        x = global_mean_pool(x, batch) 
        if return_pooled == True:
            return x
        if self.global_feature_dim > 0:
            if not hasattr(data, "global_x"):
                raise RuntimeError("global_feature_dim > 0 but batch has no global_x features.")
            global_x = data.global_x
            if global_x.dim() == 1:
                global_x = global_x.view(-1, self.global_feature_dim)
            x = torch.cat([x, global_x.to(x.device)], dim=1)
        x = self.mlp(x)
        
        return F.log_softmax(x, dim=1)
