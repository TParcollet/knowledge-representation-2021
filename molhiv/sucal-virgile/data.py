#! /bin/env python3
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from torch_geometric.loader import DataLoader

from model import *

device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
print("Device:", device)

arxiv_name = "ogbn-arxiv"
molhiv_name = "ogbg-molhiv"


def import_nodes(d_name):
    """
    Import nodes dataset

    :param d_name: Dataset name
    :return: Dataset
    """

    dataset = PygNodePropPredDataset(name=d_name)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph = dataset[0]  # pyg graph object
    return dataset, train_idx, valid_idx, test_idx


def import_graph(d_name):
    """
    Import graph dataset

    :param d_name: Dataset name
    :return: Dataset
    """

    dataset = PygGraphPropPredDataset(name=d_name)
    split_idx = dataset.get_idx_split()
    return dataset, split_idx["train"], split_idx["valid"], split_idx["test"]


def export_torch_io_data(input_data, target_data, input_name, target_name):
    """
    Export pre-processed data

    This function is deprecated.

    :param input_data: Input dataset
    :param target_data: Output dataset
    :param input_name: Input dataset name
    :param target_name: Output dataset name
    :return: Numpy arrays of datasets
    """

    input_data_numpy = input_data.detach().numpy()
    target_data_numpy = target_data.detach().numpy().T
    input_dataframe = pd.DataFrame(input_data_numpy)
    input_dataframe.to_csv(join(dirname(abspath(__file__)), input_name), index=False)
    output_dataframe = pd.DataFrame(target_data_numpy)
    output_dataframe.to_csv(join(dirname(abspath(__file__)), target_name), index=False)
    return input_data_numpy, target_data_numpy


def import_torch_io_data(input_name, target_name):
    """
    Import pre-processed data

    This function is deprecated.

    :param input_name: Input dataset name
    :param target_name: Output dataset name
    :return: Numpy arrays of datasets
    """

    input_dataframe = pd.read_csv(join(dirname(abspath(__file__)), input_name))
    output_dataframe = pd.read_csv(join(dirname(abspath(__file__)), target_name))
    input_data_numpy = input_dataframe.to_numpy()
    target_data_numpy = output_dataframe.to_numpy()
    return input_data_numpy, target_data_numpy


if __name__ == '__main__':
    verbose = True
    print_every = 1000

    atom_encoder = AtomEncoder(emb_dim=128)
    bond_encoder = BondEncoder(emb_dim=128)
    dataset, molhiv_train_idx, molhiv_valid_idx, molhiv_test_idx = import_graph(molhiv_name)

    for (split_idx, raw_input_name, raw_target_name, raw_edge_index_name) in [
        (molhiv_train_idx, train_raw_input_name, train_raw_target_name, train_raw_edge_index_name),
        (molhiv_valid_idx, valid_raw_input_name, valid_raw_target_name, valid_raw_edge_index_name),
        (molhiv_test_idx, test_raw_input_name, test_raw_target_name, test_raw_edge_index_name),
    ]:
        print('Pre-process data into "' + raw_input_name + '", "' + raw_target_name + '" and "' + raw_edge_index_name + '".')
        edge_index_dataset = []
        # atoms_dataset = []
        # bonds_dataset = []
        input_dataset = []
        y_dataset = []
        it = 0
        dataset_split = dataset[split_idx]
        size = len(dataset_split)
        for mol in dataset_split:
            it += 1
            edge_index = mol["edge_index"]
            # print(edge_index)
            # print(edge_index.size())
            edge_index_dataset.append(edge_index)
            atoms_ebeddings = torch.mean(atom_encoder(mol['x']), 0)
            # print(atoms_ebeddings)
            # print(atoms_ebeddings.size())
            # atoms_dataset.append(atoms_ebeddings)
            bonds_ebeddings = torch.mean(bond_encoder(mol['edge_attr']), 0)
            # print(bonds_ebeddings)
            # print(bonds_ebeddings.size())
            # bonds_dataset.append(bonds_ebeddings)
            input = torch.cat([atoms_ebeddings, bonds_ebeddings], 0)
            # print(input)
            # print(input.size())
            input_dataset.append(input)
            y = mol['y']
            y_dataset.append(y[0])
            if it % print_every == 0:
                print("%.2f %s" % ((it / size) * 100, "%"))

        input_data_numpy = export_torch_data(torch.stack(input_dataset), raw_input_name)
        output_data_numpy = export_torch_data(torch.stack(y_dataset), raw_target_name)

        edge_index = dict(zip([str(i) for i in range(len(edge_index_dataset))], edge_index_dataset))
        # print(edge_index)
        torch.save(edge_index, join(dirname(abspath(__file__)), raw_edge_index_name))




