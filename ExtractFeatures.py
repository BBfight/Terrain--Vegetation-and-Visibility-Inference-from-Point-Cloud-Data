import csv
import os.path
import pandas as pd

from Features import *

'''
### USAGE ###

Input:
	- out_path: path of the folder where to store the csv containing features and characteristics
	- trees_path: path of the folder containing the segmented trees, where trees are divided by areas
	- areas: number of areas in which the point cloud surface is divided
	- trees_labels: path of the csv containing the trees labels
Output:
	- the functions will store the csvs with the features and the characteristics in the out_path folder
 
IMPORTANT NOTE: given that I don't know which is gonna be the availability of labels when using this cose, I SKIPPED the part
                in which I add the labels to the csv. LABELS SHOULD BE ADDED BY PROPERLY MODIFYING THE CODE
'''

def extractFeaturesAndCharacteristics(out_path, trees_path, areas, trees_labels):
	"""This function extracts the features for the inference task using the functions implemented in the file 'Features.py'
	   and the trees characteristics used to approximate the trees for the lines of sight analysis
	"""
	for j in range(areas):
		# setting number of trees in each area
		trees_number_per_area = []
		for i in range(areas):
			trees_number_per_area.append(len([name for name in os.listdir(trees_path + "Trees Area %i" % (i+1))]))

		# getting the trees location from the csv
		locations = pd.DataFrame.from_csv(trees_locations, header=0, index_col=None)
		# opening the file with the trees labels
		labels = pd.DataFrame.from_csv(trees_labels, header=0, index_col=None)
		# adding a dummy column to mark locations
		locations['marked'] = 0

		# creating csv where to write the computed features and the trees properties
		with open(out_path + "trees_features_area_%i" % area, 'w')\
				as trees_features, \
				open("trees_characteristics_area_%i" % area, 'w') as trees_characteristics:
			# creating writer for trees features
			fieldnames_features = ['height', 'trunk_width', 'density', 'delta_x', 'delta_y', 'average_height',
								   'red_intensity', 'blue_intensity', 'green_intensity']
			writer_features = csv.DictWriter(trees_features, fieldnames=fieldnames_features)
			writer_features.writeheader()
			# creating writer for tree characteristics
			fieldnames_characteristics = ['height', 'centroid_x', 'centroid_y', 'delta_x', 'delta_y']
			writer__characteristics = csv.DictWriter(trees_characteristics, fieldnames=fieldnames_characteristics)
			writer__characteristics.writeheader()
			# cycling over trees
			for j in range(areas):
				print 'starting processing trees for area ' + str(j+1)
				for i in range(trees_number_per_area[j]):
					# opening input file and saving points in numpy array
					points = np.genfromtxt(trees_path + "Trees Area %i/tree_%i.csv" % (j+1, i+1),
										   delimiter=",", skip_header=True)[:, 0:6]
					# write computed features on csv (ADD LABELS HERE)
					writer_features.writerow({'height': height(points), 'trunk_width': trunk_width(points),
											  'density': density(points), 'delta_x': delta_x(points),
											  'delta_y': delta_y(points), 'average_height': average_height(points),
											  'red_intensity': red_intensity(points),
											  'blue_intensity': blue_intensity(points),
											  'green_intensity': green_intensity(points),})
					# write computed characteristics on csv (AND HERE)
					writer__characteristics.writerow({'height': height(points),
													  'centroid_x': np.mean(points.transpose()[0]),
													  'centroid_y': np.mean(points.transpose()[1]),
													  'delta_x': delta_x(points),
													  'delta_y': delta_y(points)})
					# printing progress on screen
					print 'processed ' + str(i+1) + '/' + str(trees_number_per_area[j]) + ' trees in area ' + str(j+1)
