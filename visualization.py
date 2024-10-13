import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch_geometric.utils import to_networkx


def visualize_graph(data):
    plt.figure(figsize=(8, 8))
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
    plt.title("A sample of Chemical Compound")
    plt.show()


def visualize_graph_embeddings(model, loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            all_embeddings.append(out.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.hstack(all_labels)
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    plt.figure(figsize=(8, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels, cmap='coolwarm', s=100)
    plt.colorbar(label="Graph Label")
    plt.title("t-SNE Visualization of Graph-Level Embeddings")
    plt.show()


def plot_confusion_matrix(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1).cpu().numpy()
            all_preds.append(pred)
            all_labels.append(data.y.cpu().numpy())
    all_preds = np.hstack(all_preds)
    all_labels = np.hstack(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Mutagenic', 'Mutagenic'],
                yticklabels=['Non-Mutagenic', 'Mutagenic'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Confusion Matrix")
    plt.show()

