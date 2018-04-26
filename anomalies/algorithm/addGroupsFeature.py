import os
import csv
import numpy as np

# this function return a dictionary of all vertices that are related to a popular group
# thr returned dictionary -> { vertex_num = [ 0, 0, 1, 0 ... ] } 1/0 if in/not in group in index i
def get_vertex_popular_dict():
    original_data = "networks.csv"
    # this is a list of the most popular groups - highest number of vertices
    popular = [591, 611, 590, 1342, 610, 1162, 1376, 1374, 1523, 1827, 612, 802, 2101, 613, 2100]
    vertex_dict = {}
    with open(original_data) as graph_file:
        # open networks.cvs - the original graph file
        graph_reader = csv.reader(graph_file)
        first_line = True
        for row_groups in graph_reader:
            # skip first line
            if first_line:
                first_line = False
                continue
            # add first vertex to the vertex dict if not already in there
            if row_groups[0] not in vertex_dict.keys():
                vertex_dict[row_groups[0]] = [0]*15
            # add second vertex to the vertex dict if not already in there
            if row_groups[1] not in vertex_dict.keys():
                vertex_dict[row_groups[1]] = [0]*15
            # if the edge is related to a group from the popular group mark vertex as popular
            if int(row_groups[4]) in popular:
                groupnum=popular.index(int(row_groups[4]))
                vertex_dict[row_groups[0]][groupnum] = 1
                vertex_dict[row_groups[1]][groupnum] = 1
    return vertex_dict


def get_graph_top_groups_appearance(graphs_directory="../../Keren's/data/undirected/whatsAppByTime"):
    # dictionary of all vertices who belongs to one of the popular groups
    popular_vertex_dict = get_vertex_popular_dict()

    bdict = {}
    # open directory and open all graphs (default - graphs by time)
    for graph_name in sorted(os.listdir(graphs_directory)):
        # dictionary for counting how many time each of the popular groups appears in each graph
        graph_file = open(graphs_directory + "/" + graph_name + "/input/" + graph_name + ".txt")
        bdict[graph_name] = [0]*15
        numv = 0
        dict_vertices = {}
        for row_time in graph_file:
            row_time = row_time.split()
            # add from-vertex to dict and count
            if row_time[0] not in dict_vertices:
                numv += 1
                dict_vertices[row_time[0]] = 1
                for i in range(15):
                    bdict[graph_name][i] += popular_vertex_dict[row_time[0]][i]

            # add to-vertex to dict and count
            if row_time[1] not in dict_vertices:
                numv += 1
                dict_vertices[row_time[1]] = 1
                for i in range(15):
                    bdict[graph_name][i] += popular_vertex_dict[row_time[1]][i]
        # add vectors together
        for i in range(15):
            bdict[graph_name][i] = float(bdict[graph_name][i]) / float(numv)

    # convert from dictionary to array - converted
    as_array = []
    for key in sorted(bdict):
        as_array.append(bdict[key])

    # log answer
    for i in range(len(as_array)):
        for j in range(len(as_array[i])):
            as_array[i][j] = np.log(as_array[i][j])
    return as_array


