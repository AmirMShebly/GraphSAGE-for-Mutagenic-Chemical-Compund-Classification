# GraphSAGE for Mutagenic Chemical Compund Classification


## The Concept of Mutagenicity
Mutations are changes in the DNA sequence of a cell. These changes can be caused by various factors such as Radiation, Chemicals, and errors in DNA replicating.
Mutagenic compounds, sometimes called carcinogens, are chemicals that can directly or indirectly damage DNA, leading to mutations. Here are some examples of these compounds:

Benzo[a]pyrene: Found in cigarette smoke and some foods.

![Benzo-a-pyrene](https://github.com/user-attachments/assets/abe538f8-4f36-40d0-88c6-0bc833893344)


Formaldehyde: Used in various industrial processes and household products.

![fermaldehyde](https://github.com/user-attachments/assets/88d95132-65cb-4ba1-a423-a37ab66b273b)


Aflatoxin B1: A toxin produced by certain fungi that can contaminate food.

![aflatoxin b1](https://github.com/user-attachments/assets/73f66f7d-f9e6-4b03-a17e-f5efe2ba515b)



Identifying and classifying mutagenic compounds is crucial for Drug Development, Chemical Safety, Environmental Protection, etc.

## Dataset
The MUTAG dataset within pytorch geometric comprises a collection of 188 chemical compounds, each represented as a graph. These graphs model the molecular structure
of the compounds, where each node represents an atom within the molecule (e.g., carbon, oxygen, nitrogen) and each edge represents a chemical bond between two atoms.
Nodes are associated with features like the atomic number, type, and other chemical properties of the atom.
Each compound in the MUTAG dataset is labeled as either mutagenic or non-mutagenic. This labeling reflects whether the compound is likely to cause mutations in DNA,
potentially leading to cancer or other health problems.

## Applying Graph Machine Learning to Mutagenicity Prediction
Graph Neural Netowrks, like GraphSAGE, are well-suited for modeling chemical structures as graphs, making them powerful tools for predicting mutagenicity.
Accurate models for mutagenicity prediction can help identify potentially dangerous compounds early on, leading to safer products, more effective drug development,
and better environmental protection.

## GraphSAGE: A Powerful Tool for Graph-based Learning

Original Paper:
[Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)

To address the challenge of mutagenic compound classification, I utilized the GraphSAGE model, a powerful and versatile graph neural network architecture. 
GraphSAGE excels at learning representations of nodes within a graph, enabling us to make predictionsbased on both the individual properties of a node and its
connections to other nodes.
GraphSAGE learns representations for nodes by aggregating information from their neighbors. It iteratively combines the features of a node with the features of its immediate neighbors, effectively capturing local graph structure.
It uses a variety of aggregation functions, such as mean, sum, or max pooling, to combine the information from a node's neighbors. This allows for flexibility in capturing different aspects of the graph structure.
The aggregation process involves learnable weights, which are adjusted during training to optimize the model's performance. This allows GraphSAGE to adapt to the specific characteristics of the graph dataset.

![gs](https://github.com/user-attachments/assets/1f29c483-d6c4-4a9d-926e-57dc4632ea2d)



