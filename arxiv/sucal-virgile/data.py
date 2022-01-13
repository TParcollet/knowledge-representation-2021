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
    return dataset, train_idx, valid_idx, test_idx


def import_graph(d_name):
    """
    Import graph dataset

    :param d_name: Dataset name
    :return: Dataset
    """

    dataset = PygGraphPropPredDataset(name=d_name)
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
    return dataset, train_loader, valid_loader, test_loader


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
    dataset, arxiv_train_idx, arxiv_valid_idx, arxiv_test_idx = import_nodes(arxiv_name)

    for idx, name in {
        arxiv_train_idx: arxiv_train_idx_name,
        arxiv_valid_idx: arxiv_valid_idx_name,
        arxiv_test_idx: arxiv_test_idx_name,
    }.items():
        export_torch_data(idx, name)

    edge_index = dataset[0]["edge_index"]

    print_every = 10000

    for (arxiv_idx, raw_input_name, raw_target_name) in [
        (arxiv_train_idx, train_raw_input_name, train_raw_target_name),
        (arxiv_valid_idx, valid_raw_input_name, valid_raw_target_name),
        (arxiv_test_idx, test_raw_input_name, test_raw_target_name),
    ]:
        print('Compute data for "' + raw_input_name + '" and "' + raw_target_name + '".')
        it = 0
        inputs = []
        outputs = []
        size = len(arxiv_idx)
        # for i in range(len(arxiv_idx)):
        for i in arxiv_idx:
            it += 1
            # print(i)
            word_embedding = dataset[0]["x"][i]
            # print(word_embedding)
            target_class = dataset[0]["y"][i]
            # print(target_class)

            inputs.append(word_embedding)
            outputs.append(target_class)

            if i % print_every == 0:
                print("%.2f %s" % ((it / size) * 100, "%"))

        input_data = torch.stack(inputs)
        print(input_data)

        output_data = torch.tensor(outputs)
        print(output_data)

        input_data_numpy = export_torch_data(input_data, raw_input_name)
        output_data_numpy = export_torch_data(output_data, raw_target_name)

    edge_index = export_torch_data(edge_index, raw_edge_index_name)
