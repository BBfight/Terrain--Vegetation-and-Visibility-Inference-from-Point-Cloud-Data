from itertools import chain
from UsefulFunctions import *
from sklearn.neighbors.kde import KernelDensity


'''
This file contains the functions used to compute the features engineered in order to
infer the visual osbtruction potential of trees

The only parameter needed is 'points', which corresponds to the points delineating
the structure of a speficif tree

'''


def height(points):
    """Compute height of a tree given its points"""
    return points[max_height_points_indices(points)[0]][2]


def trunk_width(points):
    """Compute width of a tree given its lowest points"""
    min_indices = min_height_points_indices(points)
    min_points = points[min_indices]
    while len(min_points) < 2:
        points = np.delete(points, min_indices, axis=0)
        min_points = np.vstack([min_points, points[min_height_points_indices(points)]])
    distances = list(chain.from_iterable(map(lambda x: [compute_distance(x, point, 2) for point in min_points],
                                             min_points)))
    return max(distances)


def density(points):
    """Compute density of a tree given its points"""
    kde = KernelDensity(kernel='gaussian').fit(points[:, 0:3])
    return np.mean(kde.score_samples(points[:, 0:3]))


def delta_x(points):
    """Compute the difference from max x and min x"""
    max_x = points[np.argmax(points.transpose()[0])][0]
    min_x = points[np.argmin(points.transpose()[0])][0]
    return max_x - min_x


def delta_y(points):
    """Compute the difference from max y and min y"""
    max_y = points[np.argmax(points.transpose()[1])][1]
    min_y = points[np.argmin(points.transpose()[1])][1]
    return max_y - min_y


def average_height(points):
    """Compute average height of the points in the tree"""
    return np.mean(points.transpose()[2])


def red_intensity(points):
    """compute average of red values"""
    return np.var(points.transpose()[3])


def blue_intensity(points):
    """compute average of blue values"""
    return np.var(points.transpose()[4])


def green_intensity(points):
    """compute average of green values"""
    return np.var(points.transpose()[5])
