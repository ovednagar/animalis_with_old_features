import os

import datetime
import numpy as np
# path_features_dict = "/home/oved/Documents/yeela final project/python code git/Keren's/data/undirected/"
# path_graph_directory = "/home/oved/Documents/yeela final project/python code git/Keren's/data/undirected/"


def get_graph_dictonary(path_graph_dir, graph_type):
    return generate_graphs_matrix_dictionary(path_graph_dir + graph_type, path_graph_dir + graph_type + "/",
                                          path_graph_dir + graph_type + "/")


def generate_graphs_matrix_dictionary(graph_directory, graph_dict_path, features_dict_path):
    features_for_graph_dict = {}
    sorted_by_date = sorted(os.listdir(graph_directory), key=lambda x: datetime.datetime.strptime(x, '%d-%b-%Y'))
    i = 0
    for name in sorted_by_date:
        if name in ['13-Dec-2000', '18-Oct-2001', '22-Oct-2001', '19-Nov-2001', '23-Jan-2002', '30-Jan-2002', '04-Feb-2002']:
            print (name + " - " + str(i))
        i += 1
        features_for_graph_dict[name] = create_matrix(graph_directory + "/" + name + "/features/output",
            nodes_to_dictionary(graph_directory + "/" + name + "/input/" + name + ".txt"),
            graph_dict_path + name + "/graph_dictionary.txt", features_dict_path + name + "/features_dictionary.txt")
    return features_for_graph_dict


def nodes_to_dictionary(graph_file_name):
    # open graph file
    graph_file = open(graph_file_name)
    graph_dictionary = {}
    index = 0
    for line in graph_file:
        # split row <node> <node> <weight=1>
        node1, node2, weight = line.split()
        # if vertices are not in graph dictionary the add them
        if node1 not in graph_dictionary:
            graph_dictionary[node1] = index
            index += 1
        if node2 not in graph_dictionary:
            graph_dictionary[node2] = index
            index += 1
    # return dictionary
    return graph_dictionary


def create_matrix(dir_name, graph_dict, out_graph, out_features):
    # create matrix for features per graph
    matrix = []
    # map features to index
    feature_dict = {}
    index = 0
    for file_name in sorted(os.listdir(dir_name)):
        # if feature file is empty
        if os.path.getsize(dir_name + "/" + file_name) == 0:
            continue
        # vec is list of vectors of features from same file
        vec = []
        # open features file
        feature_file = open(dir_name + "/" + file_name)
        first_row = True
        # number of features in the file
        total_features = 0
        for row in feature_file:
            # extract row
            row = row.replace(",", " ")
            row = row.split()
            # on first row - update number of features for file and update features dictionary
            if first_row:

                # strip .txt
                file_name = file_name.replace(".txt", "")
                # add first feature
                feature_dict[file_name] = index
                index += 1
                # update number of features for file
                total_features = len(row) - 1

                # add extra features to dictionary if there are any
                for i in range(1, total_features):
                    feature_dict[file_name + "_" + str(i)] = index
                    index += 1
                # init vec as length of number of vertices in graph (all initiated to 0)
                for i in range(total_features):
                        vec.append([])
                        for j in range(len(graph_dict)):
                            vec[i].append(0)
                first_row = False

            # get index of row from graph dictionary
            vertex = graph_dict[row[0]]

            for i in range(total_features):
                # add feature value to vertex
                if float(row[i + 1]) < 0.0001:
                    vec[i][vertex] = float(0.0001)
                else:
                    vec[i][vertex] = float(row[i + 1])

        # add all vectors to matrix
        for i in range(total_features):
            # check that the most of the vector is not fixed number
            matrix.append(vec[i])

    # convert to numpy matrix
    matrix = np.matrix(matrix).T

    out_graph_file = open(out_graph, "w")
    out_features_file = open(out_features, "w")

    # write to files
    for key, val in graph_dict.items():
        out_graph_file.write(key + "\t" + str(val) + "\n")

    for key, val in feature_dict.items():
        out_features_file.write(key + "\t" + str(val) + "\n")

    return np.log(matrix)
