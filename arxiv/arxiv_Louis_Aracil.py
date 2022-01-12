import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        # batch normalisation. Eviter le surapprentissage grâce à une normalisation des données.
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        # calcul la convolution sur les couches cachées
        self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        # donne les couches de sortie du model
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def forward(self, x, adj_t):
        # ajout des convolutions
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # récupération de la couche de sortie
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

# bas" sur https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    outputs = model(data.x, data.adj_t)[train_idx]
    # squeeze permet de transfomer les résultats sous forme de vecteur au lieu d'une liste de vecteurs unitaires
    target = data.y.squeeze(1)[train_idx]
    # basé sur https://pytorch.org/docs/1.9.0/generated/torch.nn.functional.nll_loss.html
    loss = F.nll_loss(outputs, target)
    loss.backward()
    optimizer.step()

    return loss.item()


# basé sur https://ogb.stanford.edu/docs/nodeprop/
def test(model, graph, split_idx, evaluator):
    model.eval()
    out = model(graph.x, graph.adj_t)
    y_true = graph.y
    y_pred = out.argmax(dim=-1, keepdim=True)
    params = ['train', 'valid', 'test']
    result_dict = []
    for param in params:
        input_dict = {"y_true": y_true[split_idx[param]], "y_pred": y_pred[split_idx[param]]}
        result_dict.append(evaluator.eval(input_dict)['acc'])
    return result_dict

dropout = 0.5
learning_rate = 0.01
num_layers = 3
hidden_channels = 256
epochs = 50
device = 'cpu'
device = torch.device(device)

dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())

data = dataset[0]

# rend le graph non orienté.
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)

split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)

model = GCN(data.num_features, hidden_channels,
            dataset.num_classes,
            dropout).to(device)

evaluator = Evaluator(name="ogbn-arxiv")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(epochs):
    loss = train(model, data, train_idx, optimizer)
    result = test(model, data, split_idx,evaluator)
    train_accuracy = result[0]
    valid_accuracy = result[1]
    test_accuracy = result[2]

    print("Epoch ", epoch + 1, "\nLoss:", round(loss, 3), " \nAccuracy on Train:", round(train_accuracy, 3),
          "\nAccuracy on Validation:", round(valid_accuracy, 3), "\nAccuracy on Test", round(test_accuracy, 3))
    print("-------------------------------------------------------------------------------------------------")