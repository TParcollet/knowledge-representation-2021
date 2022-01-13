#! /bin/env python3

import pandas as pd
import torch
from os.path import join, dirname, exists, abspath
from numpy import array
from torch.optim import SparseAdam
from torch_geometric.nn.models import Node2Vec

from train import *

device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
print("Device:", device)


def export_wordd2vec_model(embedding_model, model_name):
    """
    Export trained Word2Vec model

    :param model_name: File name
    :return: Model
    """

    print('Export model "' + model_name + '".')
    torch.save(embedding_model, join(dirname(abspath(__file__)), model_name))


def export_wordd2vec_models(embedding_models, model_name):
    """
    Export trained Word2Vec models

    :param model_name: File name
    :return: Model
    """

    print('Export model "' + model_name + '".')
    torch.save(embedding_model, join(dirname(abspath(__file__)), model_name))


def import_wordd2vec_model(model_name):
    """
    Import already trained Word2Vec model

    This function doesn't work correctly.

    :param model_name: File name
    :return: Model
    """

    print('Import model "' + model_name + '".')
    model_path = join(dirname(abspath(__file__)), model_name)
    if exists(model_path):
        embedding_model = torch.load(model_path)
        print(embedding_model)
        for id, model in embedding_model.items():
            model.eval()
        # embedding_model.eval()
        return embedding_model
    return None


def init_word2vec(edge_index, walk_length, context_size, embedding_dim=128, batch_size=32, num_negative_samples=1, optimizer_class=SparseAdam):
    """
    Initialize Word2Vec model

    :param edge_index: Edges of the graph
    :param walk_length: Size of random walks (number of nodes)
    :param context_size: Estimation of number of positive samples
    :param embedding_dim: Dimensions of the embedding
    :param batch_size: Size of batches for training
    :param num_negative_samples: Number of positive samples
    :param optimizer_class: Criterion to use to optimize gradient [Default : SparseAdam, because edges are represented by a sparse matrix]
    :return: Model, data loader to train model, criterion
    """

    edge_index = torch.tensor(edge_index)
    embedding_model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=min(walk_length, context_size) - 1,
        num_negative_samples=num_negative_samples,
        sparse=True  # True because sparse matrices are given
    ).to(device)

    loader = embedding_model.loader(batch_size=batch_size, shuffle=True)
    optimizer = optimizer_class(embedding_model.parameters())

    return embedding_model, loader, optimizer


def train_word2vec(embedding_model, loader, optimizer, print_every=100, verbose=True):
    """
    Train Word2Vec model

    :param embedding_model: Word2Vec model
    :param loader: Data loader to train model
    :param optimizer: Criterion to use to optimize gradient
    :param print_every: Print rate
    :param verbose: True if log must be printed
    :return: Model, train loss
    """

    embedding_model.train()
    loss = 0
    size = len(loader)
    # if verbose:
    #     print(size)
    i = 0
    for pos, neg in loader:
        i += 1
        optimizer.zero_grad()
        l = embedding_model.loss(pos.to(device), neg.to(device))
        l.backward()
        optimizer.step()
        loss += l
        if verbose and i % print_every == 0:
            print("%.2f %s (%s)" % ((i / size) * 100, "%", round(loss.item() / i, 2)))
    if verbose:
        print(loss.item() / i)

    return embedding_model, loss


if __name__ == '__main__':
    verbose = True

    train_input_data_numpy = import_torch_data(train_raw_input_name)
    train_target_data_numpy = import_torch_data(train_raw_target_name)
    assert len(train_input_data_numpy) == len(train_target_data_numpy)
    train_edge_index_db = torch.load(train_raw_edge_index_name)
    train_edge_index_list = [train_edge_index_db[str(i)] for i in range(len(train_input_data_numpy))]

    # embedding_models = import_wordd2vec_model(model_name)
    # print(embedding_models)
    # print(type(embedding_models))
    # # embedding_models = None
    # # print(embedding_models)

    valid_input_data_numpy = import_torch_data(valid_raw_input_name)
    valid_target_data_numpy = import_torch_data(valid_raw_target_name)
    assert len(valid_input_data_numpy) == len(valid_target_data_numpy)
    valid_edge_index_db = torch.load(valid_raw_edge_index_name)
    valid_edge_index_list = [valid_edge_index_db[str(i)] for i in range(len(valid_input_data_numpy))]

    test_input_data_numpy = import_torch_data(test_raw_input_name)
    test_target_data_numpy = import_torch_data(test_raw_target_name)
    assert len(test_input_data_numpy) == len(test_target_data_numpy)
    test_edge_index_db = torch.load(test_raw_edge_index_name)
    test_edge_index_list = [test_edge_index_db[str(i)] for i in range(len(test_input_data_numpy))]

    print_every = 1000
    embedding_models = None
    verbose = False
    if embedding_models is None:
        embedding_models = {}
        for (name, input_data_numpy, target_data_numpy, edge_index_list) in [
            ("train", train_input_data_numpy, train_target_data_numpy, train_edge_index_list),
            ("valid", valid_input_data_numpy, valid_target_data_numpy, valid_edge_index_list),
            ("test", test_input_data_numpy, test_target_data_numpy, test_edge_index_list),
        ]:
            # tmp_embedding_models = {}
            for i in range(len(input_data_numpy)):
                edge_index = edge_index_list[i]
                walk_length = 10

                embedding_model, loader, optimizer = init_word2vec(
                    edge_index,
                    walk_length,
                    walk_length,
                    embedding_dim=128,
                    batch_size=8,
                    num_negative_samples=1,
                    optimizer_class=SparseAdam
                )

                # print(embedding_model.context_size)
                train_word2vec(embedding_model, loader, optimizer, print_every=print_every, verbose=verbose)
                embedding_models[name + str(i)] = embedding_model
        export_wordd2vec_models(embedding_models, model_name)

    print_every = 10000
    for (name, input_data_numpy, target_data_numpy, edge_index_list, input_name, target_name) in [
        ("train", train_input_data_numpy, train_target_data_numpy, train_edge_index_list, train_input_name, train_target_name),
        ("valid", valid_input_data_numpy, valid_target_data_numpy, valid_edge_index_list, valid_input_name, valid_target_name),
        ("test", test_input_data_numpy, test_target_data_numpy, test_edge_index_list, test_input_name, test_target_name),
    ]:
        print('Compute data for "' + input_name + '" and "' + target_name + '".')
        inputs = []
        outputs = []
        size = len(input_data_numpy)
        for i in range(size):
            id = name + str(i)
            # print(torch.unique(edge_index_list[0]))
            node_embedding = torch.mean(torch.stack([torch.tensor(embedding_models[id](torch.tensor(idx))[0]) for idx in torch.unique(edge_index_list[i][0])]), 0)  # edge_index_list[0] because it is an undirected graph: all link are in both directions.
            # print(node_embedding)
            # print(node_embedding.size())
            atom_bound_embedding = torch.tensor(input_data_numpy[i])
            target_class = target_data_numpy[i]
            cat_embedding = torch.cat((node_embedding, atom_bound_embedding))
            # print(cat_embedding.s&ize())

            inputs.append(cat_embedding)
            outputs.append(target_class)

            if i % print_every == 0:
                print("%.2f %s ()" % ((i / size) * 100, "%"))

        train_input_data = torch.stack(inputs)
        print(train_input_data)

        train_output_data = torch.tensor(array(outputs))
        print(train_output_data)

        train_input_data_numpy = export_torch_data(train_input_data, input_name)
        train_output_data_numpy = export_torch_data(train_output_data, target_name)

