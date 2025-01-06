import torch
import warnings
import argparse
from utils import build_graph, process_data
from train import train, unseen_train
warnings.filterwarnings('ignore')


def parse_arguments():
    parser = argparse.ArgumentParser(description='train GNN for link prediction on ingredient-target bilayer network.')
    parser.add_argument('--num_epochs', default=1, type=int, help='maximum training epochs')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--wd', default=4e-4, type=float, help='weight decay')
    parser.add_argument('--lr_period', default=10, type=int, help='period for lr_scheduler')
    parser.add_argument('--lr_decay', default=0.78, type=float, help='gamma decay factor for lr')
    parser.add_argument('--ingredient_in_dim', default=2304, type=int, help='dim of pretrained ingredient embedding')
    parser.add_argument('--target_in_dim', default=1280, type=int, help='dim of pretrained target embedding')
    parser.add_argument('--in_dim', default=512, type=int, help='dim of input embedding')
    parser.add_argument('--h_dim', default=256, type=int, help='dim of hidden embedding')
    parser.add_argument('--out_dim', default=64, type=int, help='dim of output embedding')
    parser.add_argument('--cuda', default=0, type=int, help='gpu index')
    parser.add_argument('--k', default=5, type=int, help='times for negative sampling')
    parser.add_argument('--batch_size', default=5000, type=int, help='batch size for graph sampler')
    parser.add_argument('--num_layers', default=2, type=int, help='number of GAT layers in the GNN')
    parser.add_argument('--graph_struct', default=3, type=int, help='0: BipartiteNet; 1: +IngredientSim; 2: +TargetSim; 3: BilayerNet')
    parser.add_argument('--method', default=5, type=int, help='0: RD; 1: RWR; 2: DW; 3: VGAE; 4: M2V; 5: PT')
    parser.add_argument('--k_fold', default=10, type=int, help='k-fold cross validation')
    parser.add_argument('--noise_ratio', default=0, type=float, help='add noise association')
    parser.add_argument('--noise_etype', default='it', type=str, help='add noise association')
    parser.add_argument('--unseen_setting', default=0, type=int, help='0: CV1; 1:CV2(unseen ingredient); 2: CV3(unseen target)')
    args = parser.parse_args()
    return args


def main(args):
    device = f'cuda:{args.cuda}' if args.cuda >= 0 and torch.cuda.is_available() else 'cpu'
    it_edges, is_edges, ts_edges, initial_features = process_data()
    hetero_graph, rel_list = build_graph(args, it_edges, is_edges, ts_edges, initial_features, device)
    if args.unseen_setting == 0:
        train(args, hetero_graph, rel_list, device)
        # train_full_model(args, hetero_graph, rel_list, device)
    else:
        unseen_train(args, hetero_graph, rel_list, device)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
