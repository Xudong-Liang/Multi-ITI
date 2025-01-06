import torch
import dgl
import os
import pickle as pkl
import random
import numpy as np
import pandas as pd
from numpy.linalg import norm


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    dgl.seed(seed)


def read_txt(file):
    res_list = []
    with open(file, "r") as f:
        for line in f:
            parts = line.strip().split(' ')
            res_list.append([int(parts[0]), int(parts[1])])
    return res_list


def process_data():
    base_path = "./data/"
    edge_file = "edges.txt"
    target_target_file = "target_similarity.txt"
    ingredient_similarity_file = "ingredient_similarity.txt"
    initial_embeddings = "initial_features.pkl"

    # training data
    existing_edges = read_txt(os.path.join(base_path, edge_file))
    # ingredient similarity data
    ingredient_similarity = read_txt(os.path.join(base_path, ingredient_similarity_file))
    # target similarity data
    target_similarity = read_txt(os.path.join(base_path, target_target_file))
    # initial embeddings data
    feature_file = os.path.join(base_path, initial_embeddings)

    with open(feature_file, 'rb') as f:
        initial_features = pkl.load(f)

    edges = pd.DataFrame(existing_edges, columns=['source', 'target'])
    edges = edges.set_index(edges.index.astype(str))

    is_edges = pd.DataFrame(ingredient_similarity, columns=['source', 'target'])
    is_edges = is_edges.set_index(is_edges.index.astype(str))

    ts_edges = pd.DataFrame(target_similarity, columns=['source', 'target'])
    ts_edges = ts_edges.set_index(ts_edges.index.astype(str))

    return edges, is_edges, ts_edges, initial_features


def build_graph(args, edges, is_edges, ts_edges, initial_features, device):
    os.environ['DGLBACKEND'] = 'pytorch'
    train_edges_tensor = torch.from_numpy(edges.values)
    rel_list = [('ingredient', 'it', 'target'),
                ('target', 'ti', 'ingredient'),
                ('ingredient', 'is', 'ingredient'),
                ('target', 'ts', 'target')]
    graph_data = {
        rel_list[0]: (train_edges_tensor[:, 0], train_edges_tensor[:, 1]),
        rel_list[1]: (train_edges_tensor[:, 1], train_edges_tensor[:, 0])
    }
    # add ingredient similarity knowledge
    if args.graph_struct in [1, 3]:
        is_edges_tensor = torch.from_numpy(is_edges.values)
        graph_data[rel_list[2]] = (torch.cat([is_edges_tensor[:, 0], is_edges_tensor[:, 1]]),
                                   torch.cat([is_edges_tensor[:, 1], is_edges_tensor[:, 0]]))
    # add target similarity knowledge
    if args.graph_struct in [2, 3]:
        ts_edges_tensor = torch.from_numpy(ts_edges.values)
        graph_data[rel_list[3]] = (torch.cat([ts_edges_tensor[:, 0], ts_edges_tensor[:, 1]]),
                                   torch.cat([ts_edges_tensor[:, 1], ts_edges_tensor[:, 0]]))

    hetero_graph = dgl.heterograph(graph_data)

    # dgl.save_graphs('model/graph.dgl', hetero_graph)
    if args.method == 0:
        node_features = initial_features['stochastic']
    if args.method == 1:
        node_features = initial_features['rwr']
    if args.method == 2:
        node_features = initial_features['deepwalk']
    if args.method == 3:
        node_features = initial_features['vgae']
    if args.method == 4:
        node_features = initial_features['metapath2vec']
    if args.method == 5:
        node_features = initial_features['pretrained']
    hetero_graph.ndata['features'] = node_features
    hetero_graph = hetero_graph.to(device)
    return hetero_graph, rel_list


def compute_loss(pos_score, neg_score, etype):
    n_edges = pos_score[etype].shape[0]

    if n_edges == 0 or neg_score[etype].numel() == 0:
        return torch.tensor(0.0, dtype=pos_score[etype].dtype, device=pos_score[etype].device, requires_grad=True)

    return (1 - pos_score[etype].unsqueeze(1) + neg_score[etype].view(n_edges, -1)).clamp(min=0).mean()


def cos_sim(a, b):
    cos_sim = np.sum(a * b, axis=1) / (norm(a, axis=1) * norm(b, axis=1))

    return cos_sim


def remove_unseen_nodes(node_type, graph, unseen_nodes_to_remove):
    unseen_nodes = graph.nodes(node_type).cpu().numpy()
    src, dst = graph.edges(etype='it')
    mask = np.isin(unseen_nodes, unseen_nodes_to_remove)
    nodes_to_remove = unseen_nodes[mask]
    nodes_to_remove = torch.tensor(nodes_to_remove, device=graph.device)
    remove_src = []
    remove_dst = []
    if node_type == 'ingredient':
        for i in range(len(src)):
            if src[i] in nodes_to_remove:
                remove_src.append(src[i])
                remove_dst.append(dst[i])
    elif node_type == 'target':
        for i in range(len(dst)):
            if dst[i] in nodes_to_remove:
                remove_src.append(src[i])
                remove_dst.append(dst[i])

    graph = dgl.remove_nodes(graph, nodes_to_remove, ntype=node_type)
    return graph, remove_src, remove_dst

def negative_sampling(unseen_setting, edges_to_keep_src, edges_to_keep_dst, num_targets, device='cuda', batch_size=100):
    neg_edges_src = []
    neg_edges_dst = []

    dst_set = set(edges_to_keep_dst.cpu().numpy())
    edges_to_keep_src = edges_to_keep_src.to(device)

    if unseen_setting == 1:
        for src in edges_to_keep_src:
            # Batch generate candidate negative samples on GPU
            candidates = torch.randint(0, num_targets, (batch_size,), device=device)

            # Filter out existing positive target nodes
            valid_neg_dst = candidates[~torch.isin(candidates, edges_to_keep_dst.to(device))]

            if len(valid_neg_dst) > 0:
                neg_dst = valid_neg_dst[0]
            else:
                neg_dst = torch.randint(0, num_targets, (1,), device=device).item()

            neg_edges_src.append(src)
            neg_edges_dst.append(neg_dst)

    elif unseen_setting == 2:
        for dst in edges_to_keep_dst:
            # Batch generate candidate negative samples on GPU
            candidates = torch.randint(0, num_targets, (batch_size,), device=device)

            # Filter out existing positive target nodes
            valid_neg_src = candidates[~torch.isin(candidates, edges_to_keep_src.to(device))]

            if len(valid_neg_src) > 0:
                neg_src = valid_neg_src[0]
            else:
                neg_src = torch.randint(0, num_targets, (1,), device=device).item()

            neg_edges_src.append(neg_src)
            neg_edges_dst.append(dst)

    return torch.tensor(neg_edges_src, device=device), torch.tensor(neg_edges_dst, device=device)
