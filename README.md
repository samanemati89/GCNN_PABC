# GCNN - Graph Convolutional Neural Network for Post-Stroke Aphasia Classification

This repository contains code for a **Graph Neural Network (GNN)** model designed to classify **patients with post-stroke aphasia** from **healthy controls** based on their **brain connectivity graphs**.

## 🧠 Project Overview
- **Graph Structure:**
  - **Nodes** = Brain regions (from DTI)
  - **Edges** = Structural connectivity strength
  - **Node Features** = Resting-state fMRI time-series

- **Models Implemented:**
  - **Graph Convolutional Network (GCN)**
  - **Graph Isomorphism Network (GIN)**
  - **Self-Attention Graph Pooling (SAGPool)**

## 📂 Files & Structure
| File               | Description |
|--------------------|------------|
| `dataloader.py`   | Loads and preprocesses graph data |
| `model.py`        | Defines different GNN models (GCN, GIN, etc.) |
| `layer.py`        | Implements custom graph layers and pooling functions |
| `network.py`      | Implements Self-Attention Graph Pooling (SAGPool) |
| `train.py`        | Trains the model |
| `test.py`         | Evaluates the model |
| `utils.py`        | Utility functions for preprocessing and stats |
| `visualization.py`| Generates plots and visualizations |

## ⚙️ Installation
To run this project, install the required dependencies:
```bash
pip install torch dgl numpy pandas scikit-learn matplotlib
