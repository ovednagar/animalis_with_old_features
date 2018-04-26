import random
import numpy as np


# this function creates dictionary of features matrix for graphs
# this function can create anomalies in graphs and create unrealiable features
def main():

    num_of_graphs = 100
    num_of_features = 30
    bad_features = 10
    features_scope = 4
    train_dict = {}
    # number of vertices in each graph
    min_num_of_vertices = 100
    max_num_of_vertices = 250
    number_of_anomalies = 5
    # list of graph anomalies
    anomalies_list = [4, 17, 58, 77, 85]

    # creating empty matrix
    for i in range(num_of_graphs):
        train_dict[i] = [0] * int(min_num_of_vertices + random.random()*(max_num_of_vertices - min_num_of_vertices))
    for key in train_dict:
        for i in range(len(train_dict[key])):
            train_dict[key][i] = [0] * num_of_features

    # empty array to hold pivot numbers
    pivot_array = [0] * num_of_features
    # random 0-features_scope integers
    for i in range(num_of_features):
        pivot_array[i] = random.randrange(0, features_scope)

    # filling matrix
    for key in train_dict:
        for i in range(len(train_dict[key])):
            for j in range(num_of_features):
                # first features with good values (close to pivot values)
                if j < num_of_features - bad_features:
                    train_dict[key][i][j] = np.log(0.1 + 1*random.random() + pivot_array[j] - 0.5)
                # last features (bad features) with values between 0 - features scope, not close to pivot val
                else:
                    train_dict[key][i][j] = np.log(0.1 + random.random()*features_scope)

    # create random graphs - anomalies !!!
    for graph in anomalies_list:
        for i in range(len(train_dict[graph])):
            for j in range(num_of_features):
                train_dict[graph][i][j] = np.log(0.1 + random.random() * features_scope)

    # convert to numpy matrix
    for graph in train_dict:
        train_dict[graph] = np.matrix(train_dict[graph])

    return train_dict
