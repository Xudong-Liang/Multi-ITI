# Herbal ingredient-target interaction prediction via multi-modal learning (Multi-ITI)
## Introduction


Multi-ITI is a multi-modal learning framework for predicting Herbal ingredient-target interactions (ITIs). It consists of a biological feature learning module and a heterogeneous graph learning module. The biological feature learning module integrates pre-trained models to build deep feature representations for ingredients and targets, while the heterogeneous graph learning module leverages a heterogeneous graph neural network with dynamic attention mechanisms to capture ingredient-target network interactions and mitigate the impact of noisy connections.

## Environment
We conduct our experiments with python3.8. Here are the requirements
```
descriptastorus
matplotlib
networkx
numpy
pandas
prettytable
rdkit
Requests
scikit_learn
scipy
subword_nmt
torch
torch_geometric
torchvision
```

## Usage

```
python main.py
```

## Acknowledgement
DGL: [https://www.dgl.ai](https://www.dgl.ai/)
KPGT: [https://github.com/lihan97/KPGT](https://github.com/lihan97/KPGT)
ESM: [https://github.com/rish-16/aft-pytorch ](https://github.com/facebookresearch/esm) 
