from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import GCNConv
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

class GCN(torch.nn.Module):
    def __init__(self, num_tasks, emb_dim, dropout = 0.5):
        super(GCN, self).__init__()


        self.dropout = dropout
        # permet l'initialisation des embeddings des nodes pour le GCN
        self.atom_encoder = AtomEncoder(emb_dim)
        # taille de la couche de sortie
        self.num_tasks = num_tasks
        # dimension des embeddings à calculer
        self.emb_dim = emb_dim

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        # calcul de la convolution en fonction des dimensions choisies
        self.convs = GCNConv(emb_dim, emb_dim)
        # batch normalisation. Eviter le surapprentissage grâce à une normalisation des données
        self.bns = torch.nn.BatchNorm1d(emb_dim)
        # calcul les prédictions de la couche de sortie
        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):

        # récupère la représentation des noeuds dans molhiv
        x = batched_data.x
        # récupère le tenseur de liens entre les noeuds
        edge_index = batched_data.edge_index
        # récupère le tenseur des noeuds
        batch = batched_data.batch
        # récupère les embeddings de nodes pour initialiser le GCN
        encoder = self.atom_encoder(x)
        # modification des embeddings des nodes
        # calcul des convolutions.
        node_representation = self.convs(encoder, edge_index)
        # calcul de la batch normalisation
        node_representation = self.bns(node_representation)
        # applique un dropout et la fonction d'activation
        node_representation = F.dropout(F.relu(node_representation), self.dropout, training = self.training)
        # réduction de dimension en faisant la moyenne des embeddings des noeuds voisins
        pool = global_mean_pool(node_representation, batch)
        x = self.graph_pred_linear(pool)
        return x

# permet d'obtenir la loss d'une classification binaire
cls_criterion = torch.nn.BCEWithLogitsLoss()

# based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html Part 4, train the network
def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch)
        optimizer.zero_grad()
        is_labeled = batch.y == batch.y
        loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
        loss.backward()
        optimizer.step()

    return loss.item()

def test(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)['rocauc']

# Training settings
batch_size = 64
epochs = 20
dropout = 0.5
emb_dim = 300
device = "cpu"

dataset = PygGraphPropPredDataset(name = "ogbg-molhiv")
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name = "ogbg-molhiv")

train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)

model = GCN(num_tasks=dataset.num_tasks, emb_dim=emb_dim, dropout=dropout).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epochs):
    loss = train(model, device, train_loader, optimizer)
    train_accuracy = test(model, device, train_loader, evaluator)
    valid_accuracy = test(model, device, valid_loader, evaluator)
    test_accuracy = test(model, device, test_loader, evaluator)

    print("Epoch ", epoch + 1, "\nLoss:", round(loss, 3), " \nAccuracy on Train:", round(train_accuracy, 3),
          "\nAccuracy on Validation:", round(valid_accuracy, 3), "\nAccuracy on Test", round(test_accuracy, 3))
    print("-------------------------------------------------------------------------------------------------")