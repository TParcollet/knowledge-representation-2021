from model import GAT
from train import train,test
import argparse
import torch
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T

def main():

    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--dataset_name',type=str,default='ogbn-arxiv')
    parser.add_argument('--hidden_chanels',type=int,default=22)
    parser.add_argument('--num_layers',type=int,default=2)
    parser.add_argument('--dropout',type=int,default=0.36)
    parser.add_argument('--epochs',type=int,default=1500)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    hidden_chanels = args.hidden_chanels
    num_layers = args.num_layers
    dropout = args.dropout
    epochs = args.epochs
    torch.manual_seed(2020)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PygNodePropPredDataset(name=dataset_name,transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    model = GAT(data.num_features, hidden_chanels,dataset.num_classes, num_layers,dropout).to(device)
    evaluator = Evaluator(name=dataset_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_accuracy = 0
    for epoch in range(1, epochs):
                loss = train(model, data, train_idx, optimizer)
                result = test(model, data, split_idx, evaluator)
                if epoch % 10 == 0:
                    train_acc, valid_acc,test_acc = result
                    print(f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_acc:.2f}%, '
                          f'Valid: {100 * valid_acc:.2f}% '
                          f'Test: {100 * test_acc:.2f}% ')
                    if valid_acc > best_accuracy :
                        best_accuracy = valid_acc
                        torch.save(model.state_dict(), 'best-model-parameters.pt')
if __name__ == "__main__":
    main()
