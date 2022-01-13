import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import time
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from train import train,eval
from model import GNN
def main():
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--dataset_name',type=str,default='ogbg-molhiv')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--num_layers',type=int,default=5)
    parser.add_argument('--emb_dim',type=int,default=15)
    parser.add_argument('--drop_ratio',type=int,default=0.5)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    num_layer = args.num_layers
    emb_dim = args.emb_dim
    drop_ratio = args.drop_ratio
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = PygGraphPropPredDataset(name = dataset_name)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(dataset_name)
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
    model = GNN(num_tasks = dataset.num_tasks, num_layer = num_layer, emb_dim = emb_dim, drop_ratio = drop_ratio, virtual_node = False).to(device)
    model.load_state_dict(torch.load('best-model-parameters.pt'))
    train_perf = eval(model, device, train_loader, evaluator)
    valid_perf = eval(model, device, valid_loader, evaluator)
    test_perf = eval(model, device, test_loader, evaluator)
    print({'Train': train_perf['rocauc'], 'Validation': valid_perf['rocauc'], 'Test': test_perf['rocauc']})
if __name__ == "__main__":
    main()   

