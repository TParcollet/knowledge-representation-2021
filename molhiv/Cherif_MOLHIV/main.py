from ogb.graphproppred import PygGraphPropPredDataset
d_name="ogbg-molhiv"

dataset = PygGraphPropPredDataset(name = d_name) 

import torch

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

# Configuration de GPU
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:40000]
test_dataset = dataset[40000:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')


from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()



from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

# model = GCN(hidden_channels=64)

# print(model)


model = GCN(hidden_channels=16)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
    
         data.edge_index = data.edge_index.type(torch.LongTensor)
         data.edge_index = data.edge_index.to(device)
        #  batch.x.float().to(device),batch.edge_index.to(device),batch.batch.to(device)

         data.batch = data.batch.type(torch.LongTensor)
         data.batch = data.batch.to(device)
         data.x =  data.x.float()
         data.x = data.x.to(device)
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        #  data.y = data.y.type(torch.LongTensor)
        #  out = out.type(torch.LongTensor)
         data.y = data.y.to(device)
         loss = criterion(out, data.y.squeeze(1))  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()


     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         data.edge_index = data.edge_index.type(torch.LongTensor)
         data.edge_index = data.edge_index.to(device)
         data.batch = data.batch.type(torch.LongTensor)
         data.batch = data.batch.to(device)
         data.x =  data.x.float()
         data.x = data.x.to(device)
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         pred = pred.to(device)
         data.y = data.y.to(device)
         correct += int((pred == data.y.squeeze(1)).sum())  # Check against ground-truth labels.
         
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')