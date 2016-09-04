import pandas as pd


from sympy import Point, Segment, Ellipse
from UsefulFunctions import compute_distance


# setting horizon in meters
horizon = 3600


def check_visibility(a, b, ida, idb, dx, dy, obstructing_trees, network):
    """Given the locations(x,y,z) of the two individuals, check if the line of sight from a to b
        crosses obstructing elements or if a is looking elsewhere, thus not seeing b
    """

    # if distance over horizon then just return because they won't see anyway
    if compute_distance(a, b, 2) > horizon:
        return network
    # creating 2D line passing between the two individuals
    line = Segment(Point(a[0], a[1]), Point(b[0], b[1]))

    # creating filter for individuals boundaries
    x_max = obstructing_trees['centroid_x'] <= max(a[0], b[0])
    x_min = obstructing_trees['centroid_x'] >= min(a[0], b[0])
    y_max = obstructing_trees['centroid_y'] <= max(a[1], b[1])
    y_min = obstructing_trees['centroid_y'] >= min(a[1], b[1])

    # performing orientation analysis
    if a[0] >= b[0] and a[1] >= b[1] and dx >= 0 and dy >= 0 or \
       a[0] <= b[0] and a[1] <= b[1] and dx <= 0 and dy <= 0 or \
       a[0] >= b[0] and a[1] <= b[1] and dx >= 0 and dy <= 0 or \
       a[0] <= b[0] and a[1] >= b[1] and dx <= 0 and dy >= 0:
        check = False
    else:
        check = True
    # if check is still true then perform lines of sight analysis
    if check:
        # scanning all trees with visual obstruction potential in range of the two individuals
        for index, tree in obstructing_trees[x_max & x_min & y_max & y_min].iterrows():
            # if tree is lower than both individuals than skip it
            if tree['height'] < a[2] and tree['height'] < b[2]:
                continue
            # creating ellipse to approximate the tree shape
            tree_ellipse = Ellipse(Point(tree['centroid_x'], tree['centroid_y']),
                                   tree['delta_x']/2, tree['delta_y']/2)
            # checking if the ellipse intersects the line among individuals, if yes draw an edge
            if tree_ellipse.intersection(line):
                return network
        network.add_edge(ida, idb)
    return network
