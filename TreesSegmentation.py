from __future__ import division

import os
import csv
import laspy
from UsefulFunctions import *


'''
### USAGE ###

This function requires that the areas is already classified, meaning that ground and non-ground points were already identified.
The algorithm will segment one area per time given that if the area is very big it will take some time.
In this way the function call be parallelized by calling it on different areas different times

Input:
    - area_id: string that will be appended to the folder name where identified trees are stored for id purposes
	- out_path: path of the folder where to store the csv containing features and characteristics
	- area_file: path of the 'las' file corresponding to the area that must be segmented
	- distance_threshold: value in meters for the adaptive distance threshold
Output:
	- the functions will store the csv corresponding to each tree in the out_path folder in which both the oiginal RGB values and a specific color assigned to the points for visualization purposes are available

'''
def treesSegmentation(area_id, area_file, out_path, distance_threshold):
    """Function to segment individual trees in a given 'las' forested area file"""
    
    # printing starting messages
    print 'starting clustering area ' + area_id
    print 'the distance threshold is ' + str(distance_threshold)

    # path where to store clusters
    path = out_path + "/Trees Area " + area_id
    # if path doesn't exists, create it
    if not os.path.exists(path): 
        os.makedirs(path)
    
    # setting the ball radius in meters where to look for trees and local max
    ball_size = 30
    # threshold for the size of clusters
    cluster_size_threshold = 40
    # colors for clusters
    colors_dictionary = cl.cnames
    colors_keys = cl.cnames.keys()
    key_index = 0

    # opening input file
    inFile = laspy.file.File(area_file, mode="r")

    # saving points and removing duplicates
    non_ground_points = np.vstack(set(map(tuple, np.vstack([inFile.x, inFile.y, inFile.z, inFile.red,
                                                      inFile.green, inFile.blue, inFile.raw_classification]).transpose())))
    # picking the points classified as non-ground
    non_ground_points = non_ground_points[non_ground_points.transpose()[6] != 2]

    # threshold for the trees height(90% of max height in the cloud)
    height_threshold = 0.9*non_ground_points[max_height_points_indices(non_ground_points)][0][2]    

    # freeing memory
    inFile.close()
    indices = []

    # creating an empty list for the points that need to be stored
    cluster_points = []
    non_cluster_points = []
    non_ground_indices = []

    # progress index
    progress = 0
    total_points_processed = 0
    discarded_points = 0
    # total number of points to be processed
    size = len(non_ground_points)

    # initialization message
    print 'initialization completed...'

    # iterating until all points have been assigned to a cluster
    while len(non_ground_points) != 0:
        # add the point with max heights as first in the cluster
        top_index = max_height_points_indices(non_ground_points)[0]
        # adding the point to the cluster
        cluster_points.append(non_ground_points[top_index][:-1])
        # deleting the point from the original array
        non_ground_points = np.delete(non_ground_points, top_index, axis=0)
        # checking if there are still points to process
        if len(non_ground_points) != 0:
            # creating a subset of the point including the tree in question
            indices = get_points(cluster_points[0][0:3], non_ground_points[:, 0:3], ball_size, False)
        else:
            break
        # status update
        print 'identifying cluster ' + str(progress+1)
        print 'picked ' + str(indices.size) + ' points'
        subset = non_ground_points[indices]
        # adding the indices in the non ground point array
        indices = indices.reshape(indices.size, 1)
        subset = np.append(subset, indices, axis=1)
        # adding the indices of the points in the subset
        subset = np.append(subset, np.array(range(len(subset))).reshape(len(subset), 1), axis=1)
        # removing the most distant point for initialization
        non_cluster_points.append(subset[-1, :-3])
        # removing the point from the subset
        subset = np.delete(subset, -1, axis=0)
        # list to keep track of indices to be removed from the subset
        indices_tbd = []
        # iterating over all the points in the subset
        print 'processing points for cluster ' + str(progress+1)
        while len(np.delete(subset, indices_tbd, axis=0)) != 0:
            # creating a subset from the previously identified subset without the already assigned points
            subsubset = np.delete(subset, indices_tbd, axis=0)
            # picking the point with maximum height in the considered subset
            highest_points_indices = max_height_points_indices(subsubset)
            # iterating over the index of highest points in the subset
            for index in highest_points_indices:
                # storing point to be analyzed
                point = subsubset[index]
                # computing minimum distance from cluster points
                minimum_distance_cluster = get_minimum_distance_point(point, cluster_points, 2)
                # computing minimum distance from non cluster points
                minimum_distance_not_cluster = get_minimum_distance_point(point, non_cluster_points, 2)
                # if minimum distance cluster is greater than threshold, discard point
                if minimum_distance_cluster > determine_threshold(point, height_threshold, distance_threshold):
                    # discard point
                    non_cluster_points.append(point[:-3])
                    # adding index to the list of indices to be removed
                    indices_tbd.append(point[-1])
                elif minimum_distance_cluster < minimum_distance_not_cluster:
                    # add point to cluster
                    cluster_points.append(point[:-3])
                    # add index to the ones to be removed from all the points
                    non_ground_indices.append(point[-2])
                    # adding index to the list of indices to be removed
                    indices_tbd.append(point[-1])
                else:
                    # discard point
                    non_cluster_points.append(point[:-3])
                    # adding index to the list of indices to be removed
                    indices_tbd.append(point[-1])
        # remove points in cluster from the original non ground points
        non_ground_points = np.delete(non_ground_points, non_ground_indices, axis=0)
        # updating total progress index
        total_points_processed += len(cluster_points)
        # check threshold on the number of points in the cluster
        if len(cluster_points) >= cluster_size_threshold:
            # saving cluster as csv file
            fieldnames = ['x', 'y', 'z', 'red', 'green', 'blue', 'R', 'G', 'B']
            with open(path + "/tree_%i.csv" % (progress+1), 'w') as cluster:
                writer = csv.DictWriter(cluster, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(len(cluster_points)):
                    writer.writerow({'x': cluster_points[i][0], 'y': cluster_points[i][1], 'z': cluster_points[i][2],
                                     'red': cluster_points[i][3], 'green': cluster_points[i][4], 'blue': cluster_points[i][5],
                                     'R': hex_to_rgb(colors_dictionary[colors_keys[key_index]])[0],
                                     'G': hex_to_rgb(colors_dictionary[colors_keys[key_index]])[1],
                                     'B': hex_to_rgb(colors_dictionary[colors_keys[key_index]])[2]})
            # getting new color
            key_index += 1
            if key_index == 150:
                key_index = 0
            # updating good clusters counter
            progress += 1
            print 'Created Cluster ' + str(progress) + ', the size is ' + str(len(cluster_points)) +\
                  ' points, total progress: ' + str(total_points_processed), '/' + str(size)
        else:
            print 'Discarded cluster of size ' + str(len(cluster_points)) +\
                  ', total progress: ' + str(total_points_processed), '/' + str(size)
            # increment discarded counter and discarded points
            discarded_points += len(cluster_points)
        # reset the lists
        cluster_points = []
        non_cluster_points = []
        non_ground_indices = []

    # print final result
    print 'finished clustering, number of clusters created is ' + str(progress) + ', discarded a total of ' \
          + str(discarded_points) + ' noisy points'

