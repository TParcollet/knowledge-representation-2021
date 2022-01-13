#! /bin/env python3

from collections import Counter
import pandas as pd
from os.path import join, dirname, abspath
from numpy import array
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.fixes import loguniform
from imblearn.over_sampling import SMOTE
from ogb.nodeproppred import Evaluator
# from model import *

# All common names and functions are because this file is run on server.
results_name = "results.csv"
best_result_name = "best_result.csv"
train_raw_input_name = "train_arxiv_raw_input.csv"
train_raw_target_name = "train_arxiv_raw_output.csv"
valid_raw_input_name = "valid_arxiv_raw_input.csv"
valid_raw_target_name = "valid_arxiv_raw_output.csv"
test_raw_input_name = "test_arxiv_raw_input.csv"
test_raw_target_name = "test_arxiv_raw_output.csv"
raw_edge_index_name = "arxiv_raw_edge_index.csv"
arxiv_train_idx_name = "arxiv_train_idx.csv"
arxiv_valid_idx_name = "arxiv_valid_idx.csv"
arxiv_test_idx_name = "arxiv_test_idx.csv"
model_name = "node2vec_arxiv_1000.pt"
# model_name = "node2vec_arxiv.pt"
train_input_name = "train_arxiv_input.csv"
train_target_name = "train_arxiv_target.csv"
valid_input_name = "valid_arxiv_input.csv"
valid_target_name = "valid_arxiv_target.csv"
test_input_name = "test_arxiv_input.csv"
test_target_name = "test_arxiv_target.csv"
best_result_name = "best_result.csv"
arxiv_name = "ogbn-arxiv"


def export_torch_data(data, name, mode='w'):
    """
    Export torch tensors

    :param data: Tensor
    :param name: File name
    :param mode: Open mode
    :return: Numpy array of data
    """

    print('Export data in "' + name + '".')
    data_numpy = data.detach().numpy()
    dataframe = pd.DataFrame(data_numpy)
    dataframe.to_csv(join(dirname(abspath(__file__)), name), index=False, header=False, mode=mode)
    return data_numpy


def import_torch_data(name):
    """
    Import torch tensors

    :param name: File name
    :return: Numpy array of data
    """

    print('Import data from "' + name + '".')
    dataframe = pd.read_csv(join(dirname(abspath(__file__)), name), header=None)
    data_numpy = dataframe.to_numpy()
    return data_numpy


def write_csv_line(file_name, line, first=False):
    """
    Write CSV file line by line

    :param file_name: File name
    :param line: Line to write
    :param first: True if the line is the first one
    """

    file_path = join(dirname(abspath(__file__)), file_name)
    if first:
        with open(file_path, 'w') as file:
            file.write("")
    with open(file_path, 'a') as file:
        file.write(",".join(['"' + str(item) + '"' for item in line]) + "\n")


if __name__ == '__main__':
    verbose = True

    evaluator = Evaluator(name=arxiv_name)
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)

    train_input_data_numpy = import_torch_data(train_input_name)
    train_target_data_numpy = import_torch_data(train_target_name).T[0]
    assert len(train_input_data_numpy) == len(train_target_data_numpy)

    valid_input_data_numpy = import_torch_data(valid_input_name)
    valid_target_data_numpy = import_torch_data(valid_target_name).T[0]
    assert len(valid_input_data_numpy) == len(valid_target_data_numpy)

    test_input_data_numpy = import_torch_data(test_input_name)
    test_target_data_numpy = import_torch_data(test_target_name).T[0]
    assert len(test_input_data_numpy) == len(test_target_data_numpy)

    print(dict(sorted(dict(Counter(train_target_data_numpy.tolist())).items(), key=lambda i: i[1], reverse=True)))  # classes are imbalanced
    sampler = SMOTE()
    train_input_data_numpy, train_target_data_numpy = sampler.fit_resample(train_input_data_numpy, train_target_data_numpy)
    print(dict(sorted(dict(Counter(train_target_data_numpy.tolist())).items(), key=lambda i: i[1], reverse=True)))  # classes are balanced
    # size = len(dict(Counter(train_target_data_numpy.tolist())))
    # print("Number of classes:", size)  # 40

    write_csv_line(results_name, ["Model", "Parameters", "Score"], first=True)
    results = []
    max_iter = 100_000

    models = {
        GaussianNB(): {
        },
        LogisticRegression(): {
        },
        RandomForestClassifier(): {
        },
        DecisionTreeClassifier(): {
        },
        KNeighborsClassifier(): {
        },
    }
    n_iter_search = 10

    for model, hyper_parameters in models.items():
        print("Model:", model, "; Hyperparameters:", hyper_parameters)
        model.fit(train_input_data_numpy, train_target_data_numpy)
        score = model.score(valid_input_data_numpy, valid_target_data_numpy)
        print(score)
        result = [str(model).split(".")[-1].split("(")[0], hyper_parameters, score]
        write_csv_line(results_name, result)
        results.append([*result, model])

    model_class, best_params, score, best_model = max(results, key=lambda x: x[-2])
    print(model_class)
    print(best_params)
    # test_score = best_model.score(test_input_data_numpy, test_target_data_numpy)
    y_pred = best_model.predict(test_input_data_numpy)
    test_score = evaluator.eval({
        'y_true': array([[y] for y in test_target_data_numpy]),
        'y_pred': array([[y] for y in y_pred])
    })
    print(test_score)

    write_csv_line(best_result_name, ["Model", "Parameters", " Test Score"], first=True)
    write_csv_line(best_result_name, [model_class, best_params, test_score])

