import datetime

from FeturesMatrix import get_graph_dictonary
from CanculationB import main as calculateB
from AnomalyParameter import calculateParam
from addGroupsFeature import get_graph_top_groups_appearance
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.patches as mpatches


# main function to run correctly
def main():

    # initialize:
    # number of chosen features
    numPairFtrWanted = 25
    # calculate C for all graphs together/ for each graph separately
    singleC = True
    # include most popular groups vector?
    group_info = False
    # destination folder for regression pictures
    folder_name = str(numPairFtrWanted) + "ftr,singleC:" + str(singleC) + ",group info:" + str(group_info)
    # name of graph directory
    graph_type = "EnronInc"
    # path of directory
    path_graph_dir = "../../directed/"

    ftrDict = get_graph_dictonary(path_graph_dir, graph_type)
    ftr_dict_key_sort = sorted(ftrDict, key=lambda x: datetime.datetime.strptime(x, '%d-%b-%Y'))
    # import fabricated data-set for testing
    # ftrDict = TrainData.main()
    B = calculateB(ftrDict, graph_type, folder_name, numPairFtrWanted, singleC)

    #B = np.random.rand(len(ftr_dict_key_sort), 20)*5
    # get group info
    if group_info:
        B = np.asarray(B)
        add_to_B = get_graph_top_groups_appearance()
        B = np.matrix(np.hstack((B, add_to_B)))

    # calculate distance between each gra[h to its neighbors to determine anomalies
    #  returns list of tuples (Param, graph_name)
    for j in range(2, 70):
        ParamList = calculateParam(B, j)
        # split tuples to x,y axis for plotting
        graph = []
        param = []
        average_distance = 0
        i = 0
        for pair in range(len(ParamList)):
            graph.append(ParamList[pair][1])
            param.append(ParamList[pair][0])
            # calculate average of distances
            average_distance += ParamList[pair][0]
            i += 1
        average_distance /= len(ParamList)
        plot_dots(ftr_dict_key_sort, graph, param, graph_type, folder_name, average_distance, j)


def plot_dots(date, x_arr, y_arr, graph_type, folder_name, average_distance, neighbors):
    anomal = ['13-Dec-2000', '18-Oct-2001', '22-Oct-2001', '19-Nov-2001', '23-Jan-2002', '30-Jan-2002', '04-Feb-2002']
    anomal_num = [140, 330, 332, 352, 388, 393, 396]
    path = "../results/regression&distribution/" + graph_type + "/" + folder_name + "/distribution_" + str(neighbors)
    plt.scatter(x_arr, y_arr, color='mediumaquamarine', marker="d", s=10)
    plt.title("context distribution")
    plt.xlabel("Time", fontsize=10)
    # take 5 elements from x axis for display
    plt.xticks(x_arr, date[::int(len(x_arr) / 5)])
    plt.xticks(x_arr[::int(len(x_arr) / 5)])
    plt.xticks(rotation=3)
    patch = []
    for label, x, y in zip(date, x_arr, y_arr, ):
        if x in anomal_num and y >= (average_distance / 2.5):
            plt.scatter(x, y, color='red', marker="o", s=10)
            patch.append(mpatches.Patch(label=label, color='red'))
        elif x in anomal_num and y < (average_distance / 2.5):
            plt.scatter(x, y, color='green', marker="o", s=10)
            patch.append(mpatches.Patch(label=label, color='green'))
        elif y < (average_distance / 3.5):
            plt.scatter(x, y, color='black', marker="o", s=10)
            patch.append(mpatches.Patch(label=label, color='black'))
    plt.legend(handles=patch, fontsize='small', loc=2)

    plt.savefig(path)
    plt.clf()
    plt.close()


main()
