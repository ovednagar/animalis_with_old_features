from scipy import spatial
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xlsxwriter


def calculateParam(B, closeNeighbors=5):
    """
    we recieve a matrix of bk by rows (row1=b1)
    :return:list of tuples (param,vertex)
    """
    dMat = spatial.distance_matrix(B, B)
    dMat = dMat.astype(float)
    np.fill_diagonal(dMat, np.inf)  # diagonal is zeros
    dim, dim = dMat.shape
    paramList = []
    first_40 = 0
    for graphk in range(dim):
        if first_40 < 100:
            first_40 += 1
            dMat_row = np.asarray(dMat[graphk, 0:100])
        else:
            dMat_row = np.asarray(dMat[graphk, graphk - 100:graphk])
        sum = 0
        sorted_row=np.sort(dMat_row)
        for col in range(closeNeighbors):
            sum += sorted_row[col]
        param = 1 / sum
        paramList.append((param, graphk))

    workbook = xlsxwriter.Workbook('Dmat.xlsx')
    worksheet = workbook.add_worksheet()
    out = open("dMat.txt", "w")
    for i in range(dim):
        out.write(str(i) + "\t")
        for j in range(dim):
            out.write(str(dMat[i][j]) + ",")
            worksheet.write(i, j, str(np.around(dMat[i][j], decimals=4)))
        out.write("\n")
    workbook.close()
    heatmap(dim, dMat.tolist(), "graph num", "Graphs Distance", "graph_dist_single")
    return paramList


def heatmap(comparr, data, headers, title, path):
    ax = sns.heatmap(data, xticklabels=40, yticklabels=40, vmin=0, vmax=1)
    plt.title(title)
    plt.savefig(path)
    plt.close()
