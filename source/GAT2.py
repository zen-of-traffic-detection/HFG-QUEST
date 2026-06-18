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
            torch.nn.Linear(hidden_channels * num_heads + self.global_feature_dim, 128),  # 输入层
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(128, 64),  # 第一个隐藏层
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(64, 32),   # 第二个隐藏层
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(32, num_classes)  # 输出层
        )
        self.dropout = Dropout(dropout_rate)  # Dropout层

    def forward(self, data, return_pooled=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.gat1(x, edge_index))
        x = self.dropout(x)  # 在第一层之后添加 Dropout

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



# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv, global_mean_pool
# from torch.nn import Linear, Dropout

# class MultiTaskGATModel(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_classes_tool, num_classes_behavior, num_heads=2, dropout_rate=0.015):
#         super(MultiTaskGATModel, self).__init__()
        
#         # 第一阶段：工具识别
#         # GAT层用于提取前5个数据包（节点）的特征
#         self.gat_tool = GATConv(in_channels, 128, heads=num_heads, concat=True)
#         self.fc_tool = Linear(128 * num_heads, num_classes_tool)  # 三分类：恶意软件、良性流量、其他
        
#         # 第二阶段：行为识别
#         # GAT层用于提取所有数据包（节点）的特征
#         self.gat_behavior = GATConv(in_channels, 128, heads=num_heads, concat=True)
#         self.fc_behavior = Linear(128 * num_heads, num_classes_behavior)  # 三分类：恶意软件行为1、2、3
        
#         self.dropout = Dropout(dropout_rate)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         # 第一阶段：工具识别（仅使用前5个节点）
#         x_tool = x[:5]  # 仅使用前5个数据包（节点）
#         x_tool = F.relu(self.gat_tool(x_tool, edge_index))
#         x_tool = self.dropout(x_tool)  # 在第一层之后添加 Dropout
#         tool_output = self.fc_tool(x_tool)
        
#         # 第二阶段：行为识别（如果是恶意软件，则使用所有节点）
#         # 这里假设恶意软件的输出为 class 0
#         if tool_output.argmax(dim=1) == 0:  # 如果预测为恶意软件
#             x_behavior = F.relu(self.gat_behavior(x, edge_index))  # 使用所有节点
#             x_behavior = self.dropout(x_behavior)
#             behavior_output = self.fc_behavior(x_behavior)
#         else:
#             behavior_output = None  # 如果不是恶意软件，行为识别不进行
        
#         return tool_output, behavior_output
