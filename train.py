import dgl
import numpy as np
from model import Model
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import compute_loss, cos_sim
from sklearn.model_selection import KFold
from dgl.dataloading.negative_sampler import GlobalUniform
from utils import set_seed
from tqdm import tqdm
import time
from utils import remove_unseen_nodes, negative_sampling
import random
import torch
set_seed(410)


def add_unknown_edges_to_graph(hetero_graph, noise_etype, noise_ratio, device):
    src_nodes, dst_nodes = hetero_graph.edges(etype=noise_etype)
    known_edges = set(zip(src_nodes, dst_nodes))
    num_known_edges = len(known_edges)
    num_unknown_edges = int(noise_ratio * num_known_edges)
    all_possible_edges = set(zip(src_nodes, dst_nodes)) - known_edges
    unknown_edges = random.sample(all_possible_edges, num_unknown_edges)
    unknown_src_nodes, unknown_dst_nodes = zip(*unknown_edges)
    unknown_src_nodes = torch.tensor(list(unknown_src_nodes), device=device)
    unknown_dst_nodes = torch.tensor(list(unknown_dst_nodes), device=device)
    hetero_graph.add_edges(unknown_src_nodes, unknown_dst_nodes, etype=noise_etype)
    return hetero_graph


def train(args, hetero_graph, rel_list, device):
    if args.method == 0:
        print("Without pre-trained embeddings. Stochastic initializing node features from uniform distribution!")

    if args.method == 1:
        print("Without pre-trained embeddings. Initializing node features based on rwr!")
    if args.method == 2:
        print("Without pre-trained embeddings. Initializing node features based on deepwalk!")
    if args.method == 3:
        print("Without pre-trained embeddings. Initializing node features based on vgae!")
    if args.method == 4:
        print("Without pre-trained embeddings. Initializing node features based on metapath2vec!")
    if args.method == 5:
        print("With pre-trained embeddings from KPGT and esm-2!")

    it_eids = hetero_graph.edges(etype='it', form='eid')
    if args.noise_ratio != 0:
        hetero_graph = add_unknown_edges_to_graph(hetero_graph, args.noise_etype, args.noise_ratio, device)
    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=411)
    fold = 0
    results = []
    for train_idx, val_idx in kf.split(it_eids, it_eids):
        start_time = time.time()
        train_eid_dict = {'it': it_eids[train_idx]}
        val_eid_dict = {'it': it_eids[val_idx]}
        for etype in hetero_graph.etypes:
            if etype != 'it':
                train_eid_dict[etype] = hetero_graph.edges(etype=etype, form='eid')
            else:
                for e in hetero_graph.edges(etype='it', form='eid'):
                    if e not in train_eid_dict['it']:
                        if e not in val_eid_dict['it']:
                            train_eid_dict['it'] = torch.cat((train_eid_dict['it'], e.unsqueeze(0)))
        train_negative_sampler = GlobalUniform(args.k)
        val_negative_sampler = GlobalUniform(1)
        train_sampler = dgl.dataloading.NeighborSampler([args.k] * args.num_layers)
        train_sampler = dgl.dataloading.as_edge_prediction_sampler(train_sampler,
                                                                   negative_sampler=train_negative_sampler)
        train_dataloader = dgl.dataloading.DataLoader(
            hetero_graph,
            train_eid_dict,
            train_sampler,
            device=device,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0
        )

        val_sampler = dgl.dataloading.NeighborSampler([args.k] * args.num_layers)
        val_sampler = dgl.dataloading.as_edge_prediction_sampler(val_sampler, negative_sampler=val_negative_sampler)
        val_dataloader = dgl.dataloading.DataLoader(
            hetero_graph,
            val_eid_dict,
            val_sampler,
            device=device,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0
        )

        model = Model(args, rel_list).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        lr_sche = torch.optim.lr_scheduler.StepLR(opt, args.lr_period, args.lr_decay)
        loss_values = []
        for epoch in tqdm(range(args.num_epochs), desc=f"Fold-{fold + 1}"):
            model.train()
            epoch_loss = []
            for step, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(train_dataloader):
                input_features = blocks[0].srcdata['features']
                pos_score, neg_score = model(args, positive_graph, negative_graph, blocks, input_features)
                loss = compute_loss(pos_score, neg_score, rel_list[0])

                epoch_loss.append(loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step()
            lr_sche.step()
            loss_values.append(np.array(epoch_loss).mean())
        end_time = time.time()
        training_time = end_time - start_time

        model.eval()
        all_positive_results = []
        all_negative_results = []

        s_time = time.time()
        with torch.no_grad():
            for input_nodes, positive_graph, negative_graph, blocks in val_dataloader:
                ingredient_features = blocks[0].srcdata['features']['ingredient']
                target_features = blocks[0].srcdata['features']['target']

                if args.method == 5:
                    mapped_ingredient = model.mapping_ingredient(ingredient_features)
                    mapped_target = model.mapping_target(target_features)
                    mapped_features = {'ingredient': mapped_ingredient, 'target': mapped_target}
                else:
                    mapped_features = {'ingredient': ingredient_features, 'target': target_features}

                node_embeddings = model.HeteroGNN(blocks, mapped_features)

                node_embeddings = {k: v.to('cpu') for k, v in node_embeddings.items()}

                src, dst = positive_graph.edges(etype='it')
                positive_test_arr = np.vstack((src.cpu().numpy(), dst.cpu().numpy())).T
                positive_res = cos_sim(np.array(node_embeddings['ingredient'][positive_test_arr[:, 0]]),
                                       np.array(node_embeddings['target'][positive_test_arr[:, 1]]))
                all_positive_results.append(positive_res)

                src, dst = negative_graph.edges(etype='it')
                negative_test_arr = np.vstack((src.cpu().numpy(), dst.cpu().numpy())).T
                negative_res = cos_sim(np.array(node_embeddings['ingredient'][negative_test_arr[:, 0]]),
                                       np.array(node_embeddings['target'][negative_test_arr[:, 1]]))
                all_negative_results.append(negative_res)
        e_time = time.time()
        inference_time = e_time - s_time
        all_positive_results = np.concatenate(all_positive_results, axis=0)
        all_negative_results = np.concatenate(all_negative_results, axis=0)

        positive_labels = np.ones(all_positive_results.shape[0])
        negative_labels = np.zeros(all_negative_results.shape[0])

        all_scores = np.concatenate([all_positive_results, all_negative_results], axis=0)
        all_labels = np.concatenate([positive_labels, negative_labels], axis=0)

        auroc = roc_auc_score(all_labels, all_scores)

        auprc = average_precision_score(all_labels, all_scores)
        print(f"Fold-{fold + 1} - AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}, Training_time: {training_time:.3f}, Inference_time: {inference_time:.3f}")
        fold += 1
        results.append((auroc, auprc, training_time))

    results_arr = np.array(results)

    mean_auroc = np.mean(results_arr[:, 0])
    mean_auprc = np.mean(results_arr[:, 1])
    mean_time = np.mean(results_arr[:, 2])

    std_auroc = np.std(results_arr[:, 0])
    std_auprc = np.std(results_arr[:, 1])
    std_time = np.std(results_arr[:, 2])
    print(
        f"{fold}-fold average performance - AUROC: {mean_auroc:.3f} ± {std_auroc:.3f}, AUPRC: {mean_auprc:.3f} ± {std_auprc:.3f}, Time: {mean_time:.3f} ± {std_time:.3f}")


def unseen_train(args, hetero_graph, rel_list, device):

    if args.unseen_setting == 1:
        # unseen ingredients
        unseen_nodes = np.arange(1237)
        node_type = 'ingredient'
        neg_num = 2204
    elif args.unseen_setting == 2:
        # unseen targets
        unseen_nodes = np.arange(2204)
        node_type = 'target'
        neg_num = 1237

    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=411)
    folds = list(kf.split(unseen_nodes))
    results = []
    for fold in range(args.k_fold):
        model = Model(args, rel_list).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        lr_sche = torch.optim.lr_scheduler.StepLR(opt, args.lr_period, args.lr_decay)

        train_indices, val_indices = folds[fold]
        val_nodes = torch.tensor(val_indices, device=device)
        train_graph, remove_src, remove_dst = remove_unseen_nodes(node_type, hetero_graph, val_nodes.cpu().numpy())
        train_graph = train_graph.to(device)

        src, dst = hetero_graph.edges(etype='it')
        if node_type == 'ingredient':
            edges_to_keep_src = torch.tensor(remove_src, device='cuda')
            edges_to_keep_dst = torch.tensor(dst, device='cuda')
            neg_edges_src, neg_edges_dst = negative_sampling(args.unseen_setting, edges_to_keep_src, edges_to_keep_dst, neg_num)
        else:
            edges_to_keep_src = torch.tensor(src, device='cuda')
            edges_to_keep_dst = torch.tensor(remove_dst, device='cuda')
            neg_edges_src, neg_edges_dst = negative_sampling(args.unseen_setting, edges_to_keep_src, edges_to_keep_dst, neg_num)

        eid_dict = {
            etype: train_graph.edges(etype=etype, form='eid')
            for etype in train_graph.etypes}

        # Negative sampling
        negative_sampler = GlobalUniform(args.k)
        sampler = dgl.dataloading.NeighborSampler([args.k] * args.num_layers)
        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, negative_sampler=negative_sampler)

        # DataLoader
        dataloader = dgl.dataloading.DataLoader(
            train_graph,
            eid_dict,
            sampler,
            device=device,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0
        )

        # Training loop
        for epoch in tqdm(range(args.num_epochs)):
            model.train()
            for step, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(dataloader):
                # print(blocks)
                input_features = blocks[0].srcdata['features']
                pos_score, neg_score = model(args, positive_graph, negative_graph, blocks, input_features)
                loss = compute_loss(pos_score, neg_score, rel_list[0])

                opt.zero_grad()
                loss.backward()
                opt.step()
            lr_sche.step()

        with torch.no_grad():
            ingredient_features = hetero_graph.ndata['features']['ingredient']
            target_features = hetero_graph.ndata['features']['target']

            if args.method == 5:
                mapped_ingredient = model.mapping_ingredient(ingredient_features)
                mapped_target = model.mapping_target(target_features)
                mapped_features = {'ingredient': mapped_ingredient, 'target': mapped_target}
            else:
                mapped_features = {'ingredient': ingredient_features, 'target': target_features}
            block = dgl.to_block(hetero_graph)
            blocks = [block, block]
            node_embeddings = model.HeteroGNN(blocks, mapped_features)
            node_embeddings = {k: v.to('cpu') for k, v in node_embeddings.items()}

            positive_res = cos_sim(np.array(node_embeddings['ingredient'][remove_src]),
                                   np.array(node_embeddings['target'][remove_dst]))

            negative_test_arr = np.vstack((neg_edges_src.cpu().numpy(), neg_edges_dst.cpu().numpy())).T
            negative_res = cos_sim(np.array(node_embeddings['ingredient'][negative_test_arr[:, 0]]),
                                   np.array(node_embeddings['target'][negative_test_arr[:, 1]]))

        positive_labels = np.ones(len(positive_res))
        negative_labels = np.zeros(len(negative_res))

        all_scores = np.concatenate([positive_res, negative_res], axis=0)
        all_labels = np.concatenate([positive_labels, negative_labels], axis=0)

        auroc = roc_auc_score(all_labels, all_scores)

        auprc = average_precision_score(all_labels, all_scores)
        print(f"Fold-{fold + 1} - AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}")
        results.append((auroc, auprc))
    results_arr = np.array(results)

    mean_auroc = np.mean(results_arr[:, 0])
    mean_auprc = np.mean(results_arr[:, 1])

    std_auroc = np.std(results_arr[:, 0])
    std_auprc = np.std(results_arr[:, 1])
    print(
        f"{fold}-fold average performance - AUROC: {mean_auroc:.3f} ± {std_auroc:.3f}, AUPRC: {mean_auprc:.3f} ± {std_auprc:.3f}")
