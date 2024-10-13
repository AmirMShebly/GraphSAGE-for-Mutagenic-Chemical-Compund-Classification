import torch
from data_loader import load_data
from model import GraphSAGEModel
from utils import train, test
from visualization import visualize_graph, visualize_graph_embeddings, plot_confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset, train_loader, test_loader = load_data()

visualize_graph(dataset[0])

print(f'Dataset: {dataset}:')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_node_features}')
print(f'Number of classes: {dataset.num_classes}')
print(dataset[0])

model = GraphSAGEModel(dataset.num_node_features, hidden_channels=64, num_classes=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

epochs = 100
for epoch in range(epochs):
    loss = train(model, train_loader, optimizer, device)
    train_acc = test(model, train_loader, device)
    test_acc = test(model, test_loader, device)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

visualize_graph_embeddings(model, test_loader, device)
plot_confusion_matrix(model, test_loader, device)
