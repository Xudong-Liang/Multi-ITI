# Herbal ingredient-target interaction prediction via multi-modal learning (Multi-ITI)
## Introduction
![image](https://github.com/Xudong-Liang/Multi-ITI/blob/main/overview.png)

Multi-ITI is a multi-modal learning framework for predicting herbal ingredient-target interactions (ITIs). It consists of a biological feature learning module and a heterogeneous graph learning module. The biological feature learning module integrates pre-trained models to build deep feature representations for ingredients and targets, while the heterogeneous graph learning module leverages a heterogeneous graph neural network with dynamic attention mechanisms to capture ingredient-target network interactions and mitigate the impact of noisy connections.

## Environment
We conduct our experiments with python3.8. Here are the requirements
```
- torch: 1.13.1
- dgl: 1.1.1+cu116
- numpy: 1.24.4
- scikit-learn: 1.3.0
- pandas: 1.5.3
- matplotlib: 3.7.2
- rdkit: 2023.9.6
- tqdm: 4.65.0
```

## Usage

```
- Download the pre-trained node embeddings file `initial_features.pkl` (https://drive.google.com/file/d/1KqOOoh_lCJbvmBiWzmX77Oxm3kW0hJeH/view?usp=sharing) and place it in the `data` folder.
- python main.py
```

## Acknowledgement
DGL: https://www.dgl.ai/  
KPGT: https://github.com/lihan97/KPGT  
ESM: https://github.com/facebookresearch/esm
