import utm
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


from ComputeLineOfSight import check_visibility


def compute_visibility_network(baboons, timestamp, area_boundaries, baboons_path, out_path):
    """Function to compute the visibility network given the path where trajectory are stored, the number
       of baboon, a given timestamp and the area boundaries (UP, DOWN, LEFT, RIGHT is the correct order)
    """

    # creating graph instance
    network = nx.DiGraph()

    # converting boundaries to utm
    area_max = utm.from_latlon(area_boundaries[0], area_boundaries[1])[0:2]
    area_min = utm.from_latlon(area_boundaries[2], area_boundaries[3])[0:2]

    # open file with the trees features and label
    trees_characteristics = pd.DataFrame.from_csv(
        'C:/Users/Ben/Google Drive/UIC/Thesis/Data/trees_characteristics_area2.csv',
        header=0, index_col=None)
    # keeping only trees with visual obstruction potential
    obstructing_trees = trees_characteristics[trees_characteristics['see/notSee'] == 0]

    # looping for each baboon using initial point in the area range
    for i in range(1, baboons+1):
        print 'parsing trajectory of baboon ' + str(i) + ' for timestamp ' + str(timestamp)
        # opening trajectory csv
        trajectory_1 = np.genfromtxt(baboons_path + "trajectory_%i.csv" % i, delimiter=",", skip_header=True)
        # keeping only points in area range
        trajectory_1 = trajectory_1[np.where(np.logical_and(np.logical_and(trajectory_1.transpose()[0] >= area_min[0],
                                                                           trajectory_1.transpose()[0] <= area_max[0]),
                                                            np.logical_and(trajectory_1.transpose()[1] >= area_min[1],
                                                                           trajectory_1.transpose()[1] <= area_max[1])))]
        # if baboons did not pass in area boundaries continue
        if len(trajectory_1) == 0:
            print 'skipped parsing baboon ' + str(i) + ' because out of areas boundaries'
            continue
        # if timestamp not included continue
        elif timestamp not in trajectory_1.transpose()[-1]:
            print 'skipped parsing baboon ' + str(i) + ' due to absence of timestamp'
            continue
        # else add node to the network and get position and deltas for direction analysis
        elif i not in network.nodes():
            network.add_node(i)
        # compute increment on x and y to determine orientation for first baboon
        pos1 = trajectory_1[trajectory_1.transpose()[-1] == timestamp][0]
        pos1[2] = pos1[2]/1000 if not np.isnan(pos1[2]) else np.nanmean(trajectory_1.transpose()[2])/1000
        if (timestamp + 1) in trajectory_1.transpose()[-1]:
            successive = trajectory_1[trajectory_1.transpose()[-1] == timestamp + 1][0]
            dx = successive[0] - pos1[0]
            dy = successive[1] - pos1[1]
        else:
            previous = trajectory_1[trajectory_1.transpose()[-1] == timestamp - 1][0]
            dx = pos1[0] - previous[0]
            dy = pos1[1] - previous[1]
        # looping on any other baboon to check for visibility
        for j in range(1, baboons+1):
            # skip if same baboon
            if j == i:
                continue
            # opening trajectory csv
            trajectory_2 = np.genfromtxt(baboons_path + "trajectory_%i.csv" % j, delimiter=",", skip_header=True)
            # keeping only points in area range
            trajectory_2 = trajectory_2[np.where(np.logical_and(np.logical_and(trajectory_2.transpose()[0] >= area_min[0],
                                                                               trajectory_2.transpose()[0] <= area_max[0]),
                                                                np.logical_and(trajectory_2.transpose()[1] >= area_min[1],
                                                                               trajectory_2.transpose()[1] <= area_max[1])))]
            # if baboons did not pass in area boundaries continue
            if len(trajectory_2) == 0:
                print 'skipped comparison with baboon ' + str(j) + ' because out of area boundaries'
                continue
            # if baboons did not pass in area boundaries continue
            elif timestamp not in trajectory_2.transpose()[-1]:
                print 'skipped comparison with baboon ' + str(j) + ' due to absence of timestamp'
                continue
            # else add node to the network
            elif j not in network.nodes():
                network.add_node(j)
            # getting position
            pos2 = trajectory_2[trajectory_2.transpose()[-1] == timestamp][0]
            pos2[2] = pos2[2]/1000 if not np.isnan(pos2[2]) else np.nanmean(trajectory_2.transpose()[2])/1000
            # if for the given timestamp baboon passed in area boundaries, then call check visibility routine
            if timestamp in trajectory_2.transpose()[-1]:
                network = check_visibility(pos1, pos2, i, j, dx, dy, obstructing_trees, network)
            print 'checked visibility with baboon ' + str(j) + ' out of ' + str(baboons) + ' for timestamp ' + str(timestamp)
        print 'parsed trajectory of baboon ' + str(j) + ' for timestamp ' + str(timestamp)

    # QT backend
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    # drawing the visibility network computed
    nx.draw(network, arrows=True, with_labels=True, node_size=2500, width=2)
    plt.savefig(out_path + str(timestamp) + '.png')
    plt.clf()

############################################## CALLING THE PROCEDURE #########################################################

# setting area boundaries
area2_up = 0.3519
area2_down = 0.3512
area2_left = 36.921740485547865
area2_right = 36.9221
# setting path where trajectory files are stored
baboons_path = "C:/Users/Ben/Google Drive/UIC/Thesis/Data/Baboons Visibility Network/Baboons Trajectories/"
# setting out path
out_path = "C:/Users/Ben/Google Drive/UIC/Thesis/Data/Baboons Visibility Network/Visibility Networks Images/"

# calling procedure for set of timestamps
for timestamp in range(178496, 178596):
    compute_visibility_network(26, timestamp, [area2_up, area2_right, area2_down, area2_left], baboons_path, out_path)