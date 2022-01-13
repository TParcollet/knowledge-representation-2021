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


def import_wordd2vec_model(model_name):
    """
    Import already trained Word2Vec model

    :param model_name: File name
    :return: Model
    """

    print('Import model "' + model_name + '".')
    model_path = join(dirname(abspath(__file__)), model_name)
    if exists(model_path):
        embedding_model = torch.load(model_path)
        embedding_model.eval()
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
    embedding_model = Node2Vec(  # Word2Vec trained with random walks as "words" sequences
        edge_index=edge_index,  # Graph
        embedding_dim=embedding_dim,  # Dimensions of the embedding
        walk_length=walk_length,  # Size of random walks (number of nodes)
        context_size=min(walk_length, context_size) - 1,  # Estimation of number of positive samples
        num_negative_samples=num_negative_samples,  # Number of positive samples
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
    if verbose:
        print(size)
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

    embedding_model = import_wordd2vec_model(model_name)
    print(embedding_model)
    print(type(embedding_model))
    # embedding_model = None
    # print(embedding_model)

    train_input_data_numpy = import_torch_data(train_raw_input_name)
    train_target_data_numpy = import_torch_data(train_raw_target_name)
    assert len(train_input_data_numpy) == len(train_target_data_numpy)

    valid_input_data_numpy = import_torch_data(valid_raw_input_name)
    valid_target_data_numpy = import_torch_data(valid_raw_target_name)
    assert len(valid_input_data_numpy) == len(valid_target_data_numpy)

    test_input_data_numpy = import_torch_data(test_raw_input_name)
    test_target_data_numpy = import_torch_data(test_raw_target_name)
    assert len(test_input_data_numpy) == len(test_target_data_numpy)

    arxiv_train_idx = import_torch_data(arxiv_train_idx_name)
    arxiv_valid_idx = import_torch_data(arxiv_valid_idx_name)
    arxiv_test_idx = import_torch_data(arxiv_test_idx_name)

    print_every = 10
    if embedding_model is None:
        edge_index = import_torch_data(raw_edge_index_name)
        walk_length = 1000

        embedding_model, loader, optimizer = init_word2vec(
            edge_index,
            walk_length,
            len(train_target_data_numpy),
            embedding_dim=128,
            batch_size=32,
            num_negative_samples=1,
            optimizer_class=SparseAdam
        )

        print(embedding_model.context_size)
        train_word2vec(embedding_model, loader, optimizer, print_every=print_every, verbose=verbose)
        export_wordd2vec_model(embedding_model, model_name)

    print_every = 10000
    for (arxiv_idx, input_name, target_name, input_data_numpy, target_data_numpy) in [
        (arxiv_train_idx, train_input_name, train_target_name, train_input_data_numpy, train_target_data_numpy),
        (arxiv_valid_idx, valid_input_name, valid_target_name, valid_input_data_numpy, valid_target_data_numpy),
        (arxiv_test_idx, test_input_name, test_target_name, test_input_data_numpy, test_target_data_numpy),
    ]:
        print('Compute data for "' + input_name + '" and "' + target_name + '".')
        inputs = []
        outputs = []
        size = len(arxiv_idx)
        for i in range(len(arxiv_idx)):
            node_embedding = torch.tensor(embedding_model(torch.tensor(arxiv_idx[i]))[0])
            word_embedding = torch.tensor(input_data_numpy[i])
            target_class = target_data_numpy[i]
            cat_embedding = torch.cat((node_embedding, word_embedding))

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

