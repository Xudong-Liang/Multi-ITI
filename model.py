import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
from torch import nn
from dgl import function as fn

class HeteroGNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, num_layers):
        super().__init__()

        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.num_heads_in = 8
        self.num_heads_hidden = 8
        self.num_heads_out = 8
        self.drop_rate = 0.2
        if self.num_layers == 1:
            # input layer
            self.conv_in = dglnn.HeteroGraphConv({
                rel: dglnn.GATv2Conv(in_feats, out_feats // self.num_heads_in, self.num_heads_in, attn_drop=self.drop_rate)
                for rel in rel_names}, aggregate='mean')

            # batch normalization
            self.bn = nn.BatchNorm1d(out_feats)
            self.bn1 = nn.BatchNorm1d(out_feats)
        else:
            # input layer
            self.conv_in = dglnn.HeteroGraphConv({
                rel: dglnn.GATv2Conv(in_feats, hid_feats // self.num_heads_in, self.num_heads_in, attn_drop=self.drop_rate)
                for rel in rel_names}, aggregate='mean')

            # hidden layer
            self.h_layers = nn.ModuleList([
                dglnn.HeteroGraphConv({
                    rel: dglnn.GATv2Conv(self.hid_feats, self.hid_feats // self.num_heads_hidden, self.num_heads_hidden, attn_drop=self.drop_rate)
                    for rel in rel_names}, aggregate='mean')
                for _ in range(self.num_layers)])

            # output layer
            self.conv_out = dglnn.HeteroGraphConv({
                rel: dglnn.GATv2Conv(self.hid_feats, self.out_feats // self.num_heads_out, self.num_heads_out)
                for rel in rel_names}, aggregate='mean')

            # batch normalization layers
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(self.hid_feats) for _ in range(self.num_layers + 1)])  # assuming two types of nodes
            self.bns1 = nn.ModuleList([nn.BatchNorm1d(self.hid_feats) for _ in range(self.num_layers + 1)])

    def forward(self, blocks, inputs):
        h = self.conv_in(blocks[0], inputs)
        self.rel_list = list(h.keys())

        if self.num_layers == 1:
            for rel in self.rel_list:
                h[rel] = F.leaky_relu(self.bn(h[rel].view(-1, self.out_feats)))
        else:
            # Input layer
            h[self.rel_list[0]] = F.leaky_relu(self.bns[0](h[self.rel_list[0]].view(-1, self.hid_feats)))
            h[self.rel_list[1]] = F.leaky_relu(self.bns1[0](h[self.rel_list[1]].view(-1, self.hid_feats)))

            # Hidden layers
            for l in range(self.num_layers - 2):
                h = self.h_layers[l](blocks[l + 1], h)
                h[self.rel_list[0]] = F.leaky_relu(self.bns[l + 1](h[self.rel_list[0]].view(-1, self.hid_feats)))
                h[self.rel_list[1]] = F.leaky_relu(self.bns1[l + 1](h[self.rel_list[1]].view(-1, self.hid_feats)))

            # Output layer
            h = self.conv_out(blocks[self.num_layers - 1], h)

        # Reshape the final embeddings to the correct output size
        h = {k: v.view(-1, self.out_feats) for k, v in h.items()}
        return h


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, h):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['h'] = h
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return edge_subgraph.edata['score']


def feature_mapping_mlp(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, out_dim)
    )


class Model(nn.Module):
    def __init__(self, args, etypes):
        super().__init__()
        if args.method == 5:
            self.mapping_ingredient = feature_mapping_mlp(args.ingredient_in_dim, args.in_dim)
            self.mapping_target = feature_mapping_mlp(args.target_in_dim, args.in_dim)
        self.HeteroGNN = HeteroGNN(args.in_dim, args.h_dim, args.out_dim, etypes, args.num_layers)
        self.pred = ScorePredictor()

    def forward(self, args, positive_graph, negative_graph, blocks, x):
        if args.method == 5:
            mapped_ingredient = self.mapping_ingredient(x['ingredient'])
            mapped_target = self.mapping_target(x['target'])
            mapped_x = {'ingredient': mapped_ingredient, 'target': mapped_target}
        else:
            mapped_x = x

        mapped_x = self.HeteroGNN(blocks, mapped_x)
        pos_score = self.pred(positive_graph, mapped_x)
        neg_score = self.pred(negative_graph, mapped_x)

        return pos_score, neg_score