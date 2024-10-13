import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGEModel(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GraphSAGEModel, self).__init__()

        self.sage1 = SAGEConv(num_node_features, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels*2)
        self.sage3 = SAGEConv(hidden_channels*2, hidden_channels*4)
        self.sage4 = SAGEConv(hidden_channels*4, hidden_channels*8)

        self.fc1 = torch.nn.Linear(hidden_channels*8, hidden_channels*4)
        self.fc2 = torch.nn.Linear(hidden_channels*4, hidden_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.sage1(x, edge_index)
        x = torch.relu(x)
        x = self.sage2(x, edge_index)
        x = torch.relu(x)
        x = self.sage3(x, edge_index)
        x = torch.relu(x)
        x = self.sage4(x, edge_index)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = self.fc2(x)

        return torch.log_softmax(x, dim=1)
