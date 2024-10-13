import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import DataLoader

def load_data():
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG', transform=NormalizeFeatures()).shuffle()
    train_dataset = dataset[:150]
    test_dataset = dataset[150:]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    return dataset, train_loader, test_loader
