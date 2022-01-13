
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from model import SAGE


def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]

    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc,test_acc


def main(model_name, hidden_dim, num_layers, dropout, lr, epochs):
    
    # For reproduciblility
    torch.manual_seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
    
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()

    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if model_name == 'sage':
      model = SAGE(data.num_features, hidden_dim,
                     dataset.num_classes, num_layers,
                     dropout).to(device)
    else:
        print('Model Unavailable')

    evaluator = Evaluator(name='ogbn-arxiv')

    best_test_accuracy = 0
    best_valid_accuracy = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, 1 + epochs):
        loss = train(model, data, train_idx, optimizer)
        result = test(model, data, split_idx, evaluator)
        if epoch % 10 == 0:
            train_acc, valid_acc, test_acc = result

            if valid_acc > best_valid_accuracy:
              best_valid_accuracy = valid_acc
              best_test_accuracy = test_acc
              torch.save(model.state_dict(), 'best_model.pt')

            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')
            
    
    best_model = SAGE(data.num_features, hidden_dim,
                     dataset.num_classes, num_layers,
                     dropout)

    best_model.load_state_dict(torch.load('best_model.pt'))
    print(test(model, data, split_idx, evaluator)[2])


if __name__=='__main__':
    hidden_dim = 128 
    num_layers = 5 
    dropout = 0.3 
    lr = 0.001 
    epochs = 500
    main('sage', hidden_dim, num_layers, dropout, lr, epochs) #0.719