from sympy import Point, Segment, Ellipse


'''
### USAGE ###

Input:
	- a: location of the first individual containing x,y and z(expressed as list or numpy array)
	- b: location of the second individual containing x,y and z(expressed as list or numpy array)	
Output:
	- True: if their line of sight is obstructed
	- False: otherwise

'''
def check_visibility(a, b):
    """Given the locations(x,y,z) of two individuals, check if they cross obstructing elements"""
    if len(a) < 3 or len(b) < 3:
        print 'not enough coordinates...exiting now'
        exit()
    # creating 2D line passing between the two individuals using the x and y coordinates
    line = Segment(Point(a[0], a[1]), Point(b[0], b[1]))
    # open file with the centroids and the max x variation and max y variation of each tree
    trees_characteristics = pd.DataFrame.from_csv('C:/Users/Ben/Google Drive/UIC/Thesis/Data/trees_characteristics.csv',
                                                  header=0, index_col=None)
    # keeping only trees with visual obstruction potential
    obstructing_trees = trees_characteristics[trees_characteristics['see/notSee'] == 0]
    # scanning trees
    for index, tree in obstructing_trees.iterrows():
        # if the tree height is lower than both individuals we ignore it
        if tree['height'] < a[2] and tree['height'] < b[2]\
                or not min(a[0], b[0]) <= tree['centroid_x'] <= max(a[0], b[0]) \
                or not min(a[1], b[1]) <= tree['centroid_y'] <= max(a[1], b[1]):
            continue
        # creating ellipse to approximate the tree shape
        tree_ellipse = Ellipse(Point(tree['centroid_x'], tree['centroid_y']),
                               tree['delta_x']/2, tree['delta_y']/2)
        # checking if the ellipse intersects the line connecting the individuals,
		# if so, the line of sight is obstructed
        if tree_ellipse.intersection(line):
            return False
    # if no obstructing tree was found in the visual, then the visual is not obstructed
    return True
