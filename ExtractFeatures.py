import csv
import os.path
import pandas as pd

from Features import *


'''
### USAGE ###

This function extracts the features for the inference task using the functions implemented in the file 'Features.py'
and the trees characteristics used to approximate the trees for the lines of sight analysis

Input:
	- out_path: path of the folder where to store the csv containing features and characteristics
	- trees_path: path of the folder containing the segmented trees, where trees are separated by area (one folder per area)
	- areas: number of areas in which the point cloud surface is divided
	- trees_labels: path of the csv containing the trees labels
    - trees_locations: path to the csv containing the ground truth trees locations(in UTM, see sample file in Data)
    - area_ids: list containing the known ids of trees with ground truth and location in the area.
                An example for the four areas I tested is the following:
                area_ids = [list(set(range(1, 14)) & valid_ids), list(set(range(14, 35)) & valid_ids),
                            list(set(range(36, 103)) & valid_ids), list(set(range(151, 220)) & valid_ids)]
                area_ids[2].extend([220, 221])
Output:
	- the functions will store the csvs with the features and the characteristics in the out_path folder
 
'''

def extractFeaturesAndCharacteristics(out_path, trees_path, areas, trees_labels, trees_locations, area_ids):
	
    # setting counters to keep track of the number of found trees
    trees_total = 0    
    found_trees_total = 0
    found_trees_areas = [[] for i in range(areas)]

    # getting the trees location from the csv
    locations = pd.DataFrame.from_csv(trees_locations, header=0, index_col=None)
    # opening the file with the trees labels
    labels = pd.DataFrame.from_csv(trees_labels, header=0, index_col=None)
    # check which tree ids are valid(means that we have location and photo)
    valid_ids = set(locations['id']) & set(labels['id'])
    # adding a dummy column to mark locations
    locations['marked'] = 0

    # setting number of trees in each area
    trees_number_per_area = []
    for i in range(areas):
        # counting trees
        trees_number_per_area.append(len([name for name in os.listdir(trees_path + "/Clusters Area %i" % (i+1))
                                          if name.endswith(".csv")]))
        # filtering aread ids for valid ones
        area_ids[i] = list(set(area_ids[i]) & valid_ids)                                          
                                          

    # creating csv where to write the computed features and the trees properties
    with open(out_path + "/trees_features", 'w') as trees_features, \
         open(out_path + "/trees_characteristics.csv", 'w') as trees_characteristics:
        # creating writer for trees features
        fieldnames_features = ['height', 'trunk_width', 'density', 'delta_x', 'delta_y', 'average_height',
                               'red_intensity', 'blue_intensity', 'green_intensity', 'see/notSee']
        writer_features = csv.DictWriter(trees_features, fieldnames=fieldnames_features)
        writer_features.writeheader()
        # creating writer for tree characteristics
        fieldnames_characteristics = ['height', 'centroid_x', 'centroid_y', 'delta_x', 'delta_y', 'see/notSee']
        writer__characteristics = csv.DictWriter(trees_characteristics, fieldnames=fieldnames_characteristics)
        writer__characteristics.writeheader()
        # cycling over trees
        for j in range(areas):
            print 'starting processing trees for area ' + str(j+1)
            for i in range(trees_number_per_area[j]):
                # opening input file and saving points in numpy array
                points = np.genfromtxt(trees_path + "/Clusters Area %i/cluster_%i.csv" % (j+1, i+1),
                                       delimiter=",", skip_header=True)[:, 0:6]
                # getting locations in area j
                locations_area = locations.ix[locations['id'].isin([id for id in locations['id'] if id in area_ids[j]])]
                # checking if tree correspond to available locations
                tree_id = is_in_locations(locations_area, points)
                if tree_id and tree_id in valid_ids:
                    # updating list to keep track of found trees per area
                    found_trees_areas[j].append(tree_id)
                    # mark location
                    locations.set_value(locations['id'] == tree_id, 'marked', 1)
                    # write computed features on csv
                    writer_features.writerow({'height': height(points), 'trunk_width': trunk_width(points),
                                              'density': density(points), 'delta_x': delta_x(points),
                                              'delta_y': delta_y(points), 'average_height': average_height(points),
                                              'red_intensity': red_intensity(points),
                                              'blue_intensity': blue_intensity(points),
                                              'green_intensity': green_intensity(points),
                                              'see/notSee': int(labels[labels['id'] == tree_id]['see/notSee'])})
                    # write computed characteristics on csv (AND HERE)
                    writer__characteristics.writerow({'height': height(points),
                                                      'centroid_x': np.mean(points.transpose()[0]),
                                                      'centroid_y': np.mean(points.transpose()[1]),
                                                      'delta_x': delta_x(points),
                                                      'delta_y': delta_y(points),
                                                      'see/notSee': int(labels[labels['id'] == tree_id]['see/notSee'])})
                # printing progress on screen
                if (i % 100) == 0:
                   print 'processed ' + str(i+1) + '/' + str(trees_number_per_area[j]) + ' trees in area ' + str(j+1)
                
    # printing per area outcome
    for j in range(areas):
        print 'found ' + str(len(found_trees_areas[j])) + '/' + str(len(area_ids[j])) + ' trees in area ' + str(j+1)
        missing_ids = set(area_ids[j]) - set(found_trees_areas[j])
        if missing_ids:
            print 'the missing ones are ' + str(missing_ids)[5:-2]
        trees_total += len(area_ids[j])
        found_trees_total += len(found_trees_areas[j])
    # printing final outcome
    print 'found ' + str(found_trees_total) + '/' + str(trees_total) + ' trees'
