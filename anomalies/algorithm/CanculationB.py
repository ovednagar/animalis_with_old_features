import datetime
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import matplotlib.patches as mpatches
import scipy.stats as stats
import os


def pearsonCurr(x, y):
    # b = [rho: pearson coefficient,p-value: how much can we relay on the result]
    b = scipy.stats.pearsonr(x, y)
    # the higher the absolute value of rho (range- between 0 to 1) the better the result
    # the smaller the p value the better the result
    return abs(b[0]), b[1]


def get_max_from_val(vec):
    val_dict = {}
    for i in range(len(vec)):
        if vec[i] not in val_dict:
            val_dict[vec[i]] = 1
        else:
            val_dict[vec[i]] += 1
    max_from_val = 0
    for key, val in val_dict.items():
        if val > max_from_val:
            max_from_val = val

    return max_from_val / len(vec)


def top(allFtr, number=20):
    """
    :param allFtr: a matrix with features in the columns and vertices in rows (all vertices of all graphs)
    :param number: how many features we want
    :return: "best"-a list of tuples (i,j) the pairs of features we want (in which their regression is best)
    quality parameter- the way to determine with pairs are best, currently is pearson correlation
    """
    # the parameter which indicates how good the regression is
    rho = []
    # a list of tuples (i,j) the pairs of features we want
    best = []
    row, col = allFtr.shape
    # runs over all pairs of features and check their quality parameter
    for i in range(col):
        for j in range(i, col):
            # obviously pair of features can't be a feature with itself..
            if i == j or get_max_from_val(np.asarray(allFtr[:, i])) > 0.9 or get_max_from_val(np.asarray(allFtr[:, j])) > 0.9:
                continue
            # returns the quality parameter (r) and how much can we rely on the result (p), for a specific pair (i,j)
            r, p_value = pearsonCurr(allFtr[:, i], allFtr[:, j])
            rho.append([r, i, j])
    # higher the quality parameter the better the regression
    rhoSort = (sorted(rho))
    # if the user requested more pair of features than we have, very unlikely
    if number > len(rhoSort):
        print("ERROR- asked for more pairs of features than there is, took all there is")
        number = len(rhoSort)
    # take the features from the end (look for the highest quality parameters)
    end = len(rhoSort) - number
    for k in range(len(rhoSort) - 1, end - 1, -1):
        # pair is a list [quality parameter, feature i, feature j], we only need information about the features
        pair = rhoSort[k]
        # pair[0]- quality parameter
        best.append((pair[1], pair[2]))
    return best


# SCATTER PLOT DESIGN FUNCTION
def regrPlot(x_arr, y_arr, c, reg_y, title, x, y, path):
    print (path)
    """
    # x vector and y vector should be of the same size
    if x_arr.size != y_arr.size:
        print ("ERROR: vector of feature i and vector of feature j not of the same size (in regression plot)")
    plt.scatter(x_arr, y_arr, color='mediumaquamarine', marker="d", s=10)
    plt.plot(x_arr, reg_y, color='salmon', linewidth=1.5)
    # patch on the left high place:
    patchC = mpatches.Patch(color='salmon', label='y = ' + str(round(c, 4)) + 'x + e')
    patchR = mpatches.Patch(color='salmon', label='number of dots:' + str(x_arr.size))
    plt.legend(handles=[patchC, patchR], fontsize='small', loc=2)
    plt.title(title)
    plt.xlabel(x, fontsize=10)
    plt.ylabel(y, fontsize=10)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
    plt.close()
    """


# REGRESSION
def regFunc(x_np, y_np):
    """
    :param x_np: numpy array for one feature
    :param y_np:
    :return: regression line, regression coefficient
    """
    x = x_np.tolist()
    y = y_np.tolist()
    regr = linear_model.LinearRegression()
    regr.fit(np.transpose(np.matrix(x)), np.transpose(np.matrix(y)))
    # reg_y is the regression line
    reg_y = regr.predict(np.transpose(np.matrix(x)))
    # coe is the coefficient
    coe = regr.coef_[0][0]
    result = (reg_y, coe)
    return result


def CanculateCBvec(allFtr, graphFtr, ftrIndex, graph_name, folder_name, graph_type, f, t, singleCoe=True):
    """
    :param allFtr: for the cij which is over all
    :param graphFtr: defined as ftrDict[graph_name]
    :param ftrIndex: list of tuples of best pair of features- [(1,2), (3,2), (i,j)...]
    :return: returns a vector b. bijk- a scalar for every pair of features(i,j) and graph k
    **This function is for a spesific graph!
    """

    #create directories to save the data:
    if "results" not in os.listdir("../"):
        os.mkdir("../results")

    if "regression&distribution" not in os.listdir("../results/"):
        os.mkdir("../results/regression&distribution/")

    if graph_type not in os.listdir("../results/regression&distribution/"):
        os.mkdir("../results/regression&distribution/" + graph_type)

    if folder_name not in os.listdir("../results/regression&distribution/" + graph_type):
        os.mkdir("../results/regression&distribution/" + graph_type + "/" + folder_name)

    # if graph_name not in os.listdir("../results/regression&distribution/" + graph_type + "/" + folder_name):
    #     os.mkdir("../results/regression&distribution/" + graph_type + "/" + folder_name + "/" + graph_name)

    # bk is vector b of graph k
    bk = []
    for i, j in ftrIndex:
        # path to save regression picture
        path = "../results/regression&distribution/" + graph_type + "/" + folder_name + "/" + graph_name + "/regression pic  " + "(" + str(
            i) + "," + str(j) + ").png"
        # if we calculate a single c over all the graphs
        if singleCoe:
            # reg_y is regression line
            # coeij is the c
            reg_y, coeij = regFunc(allFtr[f:t, i].T, allFtr[f:t, j].T)
            regrPlot(allFtr[:, i].T, allFtr[:, j].T, coeij, reg_y, "(" + str(i) + "," + str(j) + ")",
                     "feature " + str(i), "feature " + str(j), path)
        else:
            # here we calculate c for each graph separately
            # reg_y is regression line
            # coeij is the c
            reg_y, coeij = regFunc(graphFtr[:, i].T, graphFtr[:, j].T)
            xi = np.asarray(graphFtr[:, i])
            yi = np.asarray(graphFtr[:, j])
            regrPlot(xi, yi, coeij, reg_y, "(" + str(i) + "," + str(j) + ")", "featurei", "featurej", path)

        xj = graphFtr[:, j]
        xi = graphFtr[:, i]
        # calculate bkij which is defined as the mean on the b for all the vertices of a graph
        # bijkvec is a vector of bkij for all the vertices in the graph
        bijkvec = xj - coeij * xi
        bkij = np.mean(bijkvec)
        bk.append(bkij)
    return np.asarray(bk)


def main(ftrDict, graph_type, folder_name, numPairFtrWanted=20, singleC=True):
    # initialize:
    # number of graphs
    numGraphs = len(ftrDict)
    # calculate the total number of vertices in all graphs combined
    # in other words, the sum of rows (=number of vertices) in each matrix (=graph) in ftrDict
    allVertexNum = 0
    # runs on all graphs
    vertex_for_graph = []
    for key in sorted(ftrDict, key=lambda x: datetime.datetime.strptime(x, '%d-%b-%Y')):
        vertex_for_graph.append(allVertexNum)

        graph = ftrDict[key]
        # the rows of the matrix represents the number of vertices in a graph
        # the number of columns is the number of features
        vertexNumofGraph, ftrNum = graph.shape
        allVertexNum += vertexNumofGraph


    # combining all matrices in ftrDict to one matrix (allFtr)
    # not all graphs are of the same size, therefor next_empty_row indicates the next available row
    next_empty_row = 0
    # the size of the matrix is sum of all vertices over number of features
    allFtr = np.zeros((allVertexNum, ftrNum))
    # for key in sorted(ftrDict)
    for key in sorted(ftrDict, key=lambda x: datetime.datetime.strptime(x, '%d-%b-%Y')):
        # current matrix to add
        graphFtr = ftrDict[key]
        num_of_vertices_in_graph, ftrNum = graphFtr.shape
        allFtr[next_empty_row:(next_empty_row + num_of_vertices_in_graph), :] = graphFtr
        # updating the next available line
        next_empty_row = next_empty_row + num_of_vertices_in_graph

    # currRow is the index to add the bk (vector b of graph k as a row in the matrix B )
    currRow = 0
    # matrix in which row k has the b vector of graph k
    B = np.zeros((numGraphs, numPairFtrWanted))
    # chooses the best features (numPairFtrWanted is the user choice)
    # best is a list of tuples (feature i, feature j)
    #best = top(allFtr, numPairFtrWanted)
    # calculate the b vector for each graph
    from_vertex = 0
    to_vertex = vertex_for_graph[99]
    best = top(allFtr, numPairFtrWanted)
    for graph_name in sorted(ftrDict, key=lambda x: datetime.datetime.strptime(x, '%d-%b-%Y')):
        if currRow < 100:
            bk = CanculateCBvec(allFtr, ftrDict[graph_name], best, graph_name, folder_name,graph_type, from_vertex, to_vertex, singleC)
        else:
            # added
            from_vertex = vertex_for_graph[currRow - 100]
            to_vertex = vertex_for_graph[currRow]
            bk = CanculateCBvec(allFtr, ftrDict[graph_name], best, graph_name, folder_name, graph_type, from_vertex, to_vertex, singleC)
        # insert vector bk- vector b of graph k as a row in the matrix
        B[currRow, :] = bk
        currRow += 1
    return B
