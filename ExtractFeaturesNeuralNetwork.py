from __future__ import division

import pandas as pd

from UsefulFunctions import *
from SdA_mod import *


'''
### USAGE ###

Input:
	- out_path: path of the folder where to store the csv containing features and characteristics
	- input_size: number of trees to consider as input
	- hidden_size: size of the hidden layers expressed as list
	  (i.e if you want 3 hidden layers of sizes 10, 20, 30 you type [10,20,30])
	- number_of_features: number of features to be extracted by the nerual network. This will become the last hidden layer.
	  (i.e. if in the previous step you chose the 3 hidden layers and now you choose 40, overall you'll have [10, 20, 30, 40])
	- training_epochs: number of epochs used to train the nerual network
	- areas: number of areas in which the point cloud surface is divided
	- trees_path: path of the folder containing the segmented trees
	- trees_labels: path of the csv containing the trees labels
Output:
	- the functions will store the csvs with the features and the characteristics in the out_path folder
'''


# THE NEURAL NETWORK MUST BE CHANGED IN ORDER TO BE TRAINED JUST ONE TIME AND NOT EVERY TIME
# THE PART REALTIVE TO THE TREE CHARACTERISTICS MUST BE ADDED
def extractFeaturesNeuralNetwork(out_path, input_size, hidden_sizes, number_of_features, training_epochs, areas, trees_path, trees_labels):
	"""Functions to extract features from the trees point structure in an automatic way 
	   by menas of an stacked denoising autoencoder neural network
	"""
	# set minimum to initial value
	minimum = sys.maxint
	# corruption levels for the network
	corruption_levels = [.1, .2, .3, .33, .36, .39]
	# getting the trees location from the csv
	locations = pd.DataFrame.from_csv(trees_locations, header=0, index_col=None)
	# opening the file with the trees labels
	labels = pd.DataFrame.from_csv(trees_labels, header=0, index_col=None)
	# adding a dumb column to mark locations
	locations['marked'] = 0

	# setting number of trees in each area
	trees_number_per_area = []
	for i in range(areas):
		trees_number_per_area.append(len([name for name in os.listdir(trees_path + "Trees Area %i" % (j+1))]))

	# finding minimum length tree
	for j in range(areas):
		print 'starting scanning trees for area ' + str(j+1)
		for i in range(trees_number_per_area[j]):
			# opening input file
			points = np.genfromtxt(trees_path + "Trees Area %i/tree_%i.csv" % (j+1, i+1) % (j+1, i+1),
								   delimiter=",", skip_header=True)[:, 0:6]
			if len(points) < minimum:
				minimum = len(points)
			# printing progress on screen
			print 'processed ' + str(i+1) + '/' + str(trees_number_per_area[j]) + ' trees in area ' + str(j+1)

	# printing progress
	print 'Started retrieving the train data'
	# building training set
	train = []
	# sets for minimum and maximum
	x = set()
	y = set()
	z = set()
	red = set()
	blue = set()
	green = set()

	# adding values to the train
	for j in range(areas):
		for i in range(trees_number_per_area[j]):
			# opening input file
			points = np.genfromtxt(trees_path + "Trees Area %i/tree_%i.csv" % (j+1, i+1),
								   delimiter=",", skip_header=True)[:, 0:6]
			locations_area = locations.ix[locations['id'].isin([id for id in locations['id'] if id in area_ids[j]])]
			# checking if the tree is associated to a known location
			tree_id = is_in_locations(locations_area, points)
			# checking if a tree has been found and if it is in the valid set
			if not tree_id or not(tree_id in valid_ids):
				# adding values to set
				x.update(set(points.transpose()[0]))
				y.update(set(points.transpose()[1]))
				z.update(set(points.transpose()[2]))
				red.update(set(points.transpose()[3]))
				blue.update(set(points.transpose()[4]))
				green.update(set(points.transpose()[5]))
				# if more than minimum number of points then we sample
				if len(points) > minimum:
					points = points[np.random.randint(len(points), size=minimum)]
				# adding 1D array containing all points to the train set
				train.append(points.flatten())
				# decrementing n
				n -= 1
				if n == 0:
					break
		if n == 0:
			break
	# printing progress
	print 'Training data retrieved'

	# initializing list where to store max and min
	max_min = []
	# max and min value for sets
	max_min.append([max(x), min(x)])
	max_min.append([max(y), min(y)])
	max_min.append([max(z), min(z)])
	max_min.append([max(red), min(red)])
	max_min.append([max(blue), min(blue)])
	max_min.append([max(green), min(green)])

	# printing progress
	print 'Normalizing the train data'
	# normalizing train with min max paradigm
	feature_counter = 0
	for i in range(len(train)):
		for j in range(len(train[i])):
			# check if feature counter must be reset
			if feature_counter == 6:
				feature_counter = 0
			# normalize value
			train[i][j] = (max_min[feature_counter][0]-train[i][j])/(max_min[feature_counter][0]-max_min[feature_counter][1])
			# increment feature counter
			feature_counter += 1

	# converting train list to multi dimensional numpy array
	train = np.asarray(train, dtype=np.ndarray)

	# training stacked autoencoders and saving trained network
	sda = train_SdA(train=train, minimum=minimum, hidden_sizes=hidden_sizes,
				  corruption_levels=corruption_levels, n_features=n_features, pretraining_epochs=training_epochs)
	
	# printing progress
	print 'Started retrieving the test data'
	
	# generating features with the neural network for each area
	for j in range(areas):
		# creating empty list for test
		test = []
		# retrieving test data
		print 'starting processing trees for area ' + str(j+1)
		for i in range(trees_number_per_area[j]):
			# saving points in numpy array
			points = np.genfromtxt(trees_path + "Trees Area %i/tree_%i.csv" % (j+1, i+1),
								   delimiter=",", skip_header=True)[:, 0:6]
			# if more than minimum number of points then we sample
			if len(points) > minimum:
				points = points[np.random.randint(len(points), size=minimum)]
			# adding points to the test list
			test.append(points.flatten())
			# printing progress on screen
			print 'processed ' + str(i+1) + '/' + str(trees_number_per_area[j]) + ' trees in area ' + str(j+1)

		# printing progress
		print 'Normalizing the test data'

		# normalizing test with min max paradigm
		feature_counter = 0
		for i in range(len(test)):
			for j in range(len(test[i])):
				# check if feature counter must be reset
				if feature_counter == 6:
					feature_counter = 0
				# normalize value
				test[i][j] = (max_min[feature_counter][0]-test[i][j])/(max_min[feature_counter][0]-max_min[feature_counter][1])
				if test[i][j] > 1 or test[i][j] < 0:
					test[i][j] = 1
				# increment feature counter
				feature_counter += 1

		# converting test list to multi dimensional numpy array
		test = np.asarray(test, dtype=np.ndarray)
		
		# testing the neural network in the retrieved test data
		result = test_SdA(network=sda, test=test)
		
		# creating csv where to write the computed features
		trees_features = open(out_path + "trees_features_area_%i.csv" % (j+1), 'w')
				# creating csv where to write the computed features and the trees properties
		with open("trees_characteristics_area_%i" % area, 'w') as trees_characteristics:
			# creating writer for tree characteristics
			fieldnames_characteristics = ['height', 'centroid_x', 'centroid_y', 'delta_x', 'delta_y']
			writer_characteristics = csv.DictWriter(trees_characteristics, fieldnames=fieldnames_characteristics)
			writer_characteristics.writeheader()
			print 'starting processing trees for area ' + str(j+1)
			for i in range(len(result[0])):
			# opening input file
			points = np.genfromtxt(trees_path + "Trees Area %i/tree_%i.csv" % (j+1, i+1),
								   delimiter=",", skip_header=True)[:, 0:6]
			# write computed characteristics on csv
			writer_characteristics.writerow({'height': height(points),
											 'centroid_x': np.mean(points.transpose()[0]),
											 'centroid_y': np.mean(points.transpose()[1]),
										     'delta_x': delta_x(points),
										     'delta_y': delta_y(points)})
			trees_features.write(str(result[0][i].tolist())[1:-1])
			trees_features.write('\n')
		trees_features.close()
		