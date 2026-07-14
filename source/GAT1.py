import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear, Dropout

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_heads=2, dropout_rate=0.015):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, 64, heads=num_heads, concat=True)
        self.gat2 = GATConv(64 * num_heads, 64, heads=num_heads, concat=True)
        self.fc = Linear(64 * num_heads, num_classes)  
        self.dropout = Dropout(dropout_rate)

    def forward(self, data, return_pooled=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.gat1(x, edge_index))
        x = self.dropout(x)

        x = F.relu(self.gat2(x, edge_index))

        x = global_mean_pool(x, batch) 
        if return_pooled == True:
            return x
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)