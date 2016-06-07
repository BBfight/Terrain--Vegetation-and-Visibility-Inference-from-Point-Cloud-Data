import numpy as np
import math
import sklearn.neighbors as skn
import matplotlib.colors as cl

'''
This file contains a set of useful functions used to perform each step of the framework,
each function is described in details trhough proper documentation
'''

# set neighbours threshold for max and min when querying neighbours to get the tree structure
minimum_neighbours = 1000
maximum_neighbours = 5000


def compute_distance(a, b, dimensions):
    """Function to compute the distance between two points of the cloud
     in the specified number of dimensions"""
    distance = 0
    for i in range(dimensions):
        distance += (a[i]-b[i])**2
    return math.sqrt(distance)


def get_minimum_distance_point(point, points, dimensions):
    """Function to find the point with minimum distance from the points in the cloud
     in the specified number of dimensions """
    distances = map(lambda x: compute_distance(point, x, dimensions), points)
    return compute_distance(point, points[np.argmin(distances)], 2)


def get_maximum_distance_point(point, points, dimensions):
    """Function to find the point with maximum distance from the points in the cloud
    in the specified number of dimensions """
    distances = map(lambda x: compute_distance(point, x, dimensions), points)
    return compute_distance(point, points[np.argmax(distances)], 2)


def max_height_points_indices(points):
    """Function to find the indices of the points with maximum height in the cloud"""
    # if list has size 1, return index zero
    if len(points) == 1:
        return [0]
    heights = points.transpose()[2]
    max_height = heights.argmax()
    indices = np.where(heights == heights[max_height])[0]
    return indices.tolist()


def min_height_points_indices(points):
    """Function to find the indices of the points with minimum height in the cloud"""
    # if list has size 1, return index zero
    if points.size == 1:
        return [0]
    heights = points.transpose()[2]
    min_height = heights.argmin()
    indices = np.where(heights == heights[min_height])[0]
    return indices.tolist()


def determine_threshold(point, height_threshold, distance_threshold):
    """Function to set the distance threshold according to the height of the point"""
    if point[2] > height_threshold:
        return distance_threshold
    else:
        return distance_threshold-0.5


def get_points(point, points, radius, check):
    """Function to get all the indices of points within radius in points from passed point in a sorted way"""
    # if less than a certain number points left and we are not checking the local max return points indices
    if len(points) <= minimum_neighbours and not check:
        return np.nonzero(points[:, 0])[0]
    # store radius
    new_radius = radius
    tree = skn.KDTree(points, metric='euclidean')
    # getting the neighbours within radius
    neighbours = tree.query_radius(point, new_radius, True, False, True)[0][0]
    if check:
        return neighbours
    else:
        while True:
            # check if zero neighbours found, if yes increase the radius
            if neighbours.size < minimum_neighbours:
                new_radius += 1.5
            # else check if too many neighbours retrieved, in that case reduce the radius
            elif neighbours.size > maximum_neighbours:
                new_radius *= 2/3
            else:
                return neighbours
            neighbours = tree.query_radius(point, new_radius, True, False, True)[0][0]


def hex_to_rgb(hex_string):
    """function to convert hex string to rgb value"""
    rgb = cl.hex2color(hex_string)
    return tuple([int(255*x) for x in rgb])


def is_in_locations(locations, points):
    """Check if a set of points corresponds to a tree location"""
    for location in locations.iterrows():
        # if location already found continue iterating
        if location[1][3] == 1:
            continue
        # storing location point
        location_point = np.array((location[1][1], location[1][2]))
        # getting area boundaries
        max_x = np.max(points.transpose()[0])
        min_x = np.min(points.transpose()[0])
        max_y = np.max(points.transpose()[1])
        min_y = np.min(points.transpose()[1])
        # if location is in the tree boundaries than return it
        if min_x <= location_point[0] <= max_x and min_y <= location_point[1] <= max_y:
                    return location[1][0]
    return None
