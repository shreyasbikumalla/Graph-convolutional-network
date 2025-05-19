# 🧠 Node Classification using Graph Convolutional Networks (GCN)

This project implements a **Graph Convolutional Network (GCN)** to perform node classification on structured graph data. The model was developed using **PyTorch** and **PyTorch Geometric**, focusing on improving accuracy through careful preprocessing, architecture design, and hyperparameter tuning.

## 🧩 Project Overview

The goal is to classify nodes in a graph using both their feature vectors and the structure of the graph (captured in the adjacency matrix). This task is common in social networks, citation graphs, and recommendation systems.

### Key Features:
- Used real graph data represented by adjacency, feature, and label files
- Applied train/validation/test split based on JSON definitions
- Achieved **~83.7% accuracy** on the test set using fine-tuned GCN

## 📁 Files in the Repository

```
.
├── project.ipynb              # Main implementation notebook
├── adj.npz                    # Adjacency matrix of the graph
├── labels.npy                 # Node labels for classification
├── features.npy               # Feature matrix (if applicable)
├── splits.json                # Predefined train/test split
├── submission.txt             # Final submission with predicted labels
├── Project_report.pdf         # Final project writeup and explanation
├── Project_requirement.pptx   # Problem statement and evaluation criteria
└── README.md                  # This file
```

## 🛠 Technologies Used

- Python 3
- PyTorch
- PyTorch Geometric
- NumPy
- JSON for dataset splits

## 🧠 Model Architecture

- **Layer 1**: Graph Convolution + ReLU
- **Layer 2**: Graph Convolution + ReLU + Dropout
- **Layer 3**: Graph Convolution + Log Softmax

Loss: Negative Log Likelihood (NLL)  
Optimizer: Adam (lr=0.001, weight_decay=0.005)

## 🔧 How to Run

1. Install dependencies:
```bash
pip install torch torch-geometric numpy
```

2. Make sure the following data files are in the same folder:
   - `adj.npz`, `labels.npy`, `features.npy`, `splits.json`

3. Run the notebook:
   - Open `project.ipynb` in Jupyter or Colab
   - Execute all cells to train the GCN and generate predictions

## 📊 Evaluation

The model was evaluated on a test split (defined in `splits.json`) and produced a consistent **~83.7% accuracy**, confirming its generalization on unseen data.

## 🚀 Future Work

- Try deeper GNN variants (e.g., GAT, GraphSAGE)
- Integrate attention mechanisms
- Augment data or add synthetic nodes
- Apply to other domains like bioinformatics or finance

