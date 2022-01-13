import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch.nn import Linear

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

# from logger import Logger

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')


class OGBNDataset(object):
    def __init__(self, dataset_name="ogbn-arxiv"):
        self.dataset_name = dataset_name

        self.dataset = PygNodePropPredDataset(name='ogbn-arxiv',transform=T.ToSparseTensor())
        self.evaluator = Evaluator(name=self.dataset_name)
        self.args = self.parse_arg()
        self.logger = Logger(self.args.runs, self.args)
        self.device = f'cuda:{self.args.device}' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)

        self.data = self.dataset[0]
        self.data = self.data.to(self.device)

        self.split_idx = self.dataset.get_idx_split()
        self.train_idx, self.valid_idx, self.test_idx = self.split_idx["train"], self.split_idx["valid"], self.split_idx["test"]
        # self.device = self.device_GPU_CPU()

    def parse_arg(self):
        parser = argparse.ArgumentParser(description=self.dataset_name)
        parser.add_argument('--device', type=int, default=0)
        parser.add_argument('--log_steps', type=int, default=1)
        parser.add_argument('--use_sage', action='store_true')
        parser.add_argument('--num_layers', type=int, default=3)
        parser.add_argument('--hidden_channels', type=int, default=256)
        parser.add_argument('--dropout', type=float, default=0.3)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--epochs', type=int, default=300)
        parser.add_argument('--runs', type=int, default=10)
        args = parser.parse_args()
        return args 
    

    def gcn_normalizer(self):
        # Pre-compute GCN normalization.
        adj_t = self.data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        self.data.adj_t = adj_t

        self.data = self.data.to(self.device)
        return self.data


