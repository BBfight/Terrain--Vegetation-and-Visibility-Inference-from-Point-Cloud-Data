from __future__ import division


import csv
import pandas as pd


from SdA_mod import *
from Features import *
from UsefulFunctions import *


'''
### USAGE ###

This function finds the number of the smallest tree among all the trees in "trees_path"

Input:
	- areas: number of areas that you want to consider for the search
	- trees_path: path of the folder containing the segmented trees
Output:
    - returns the minimum number of points for the smallest tree
'''
def findMinimum(trees_path, areas):
    
    # setting number of trees in each area
    trees_number_per_area = []
    for i in range(areas):
        trees_number_per_area.append(len([name for name in os.listdir(trees_path + "/Clusters Area %i" % (i+1))
                                          if name.endswith(".csv")]))    
    
    # set minimum to initial value
    minimum = sys.maxint
    # finding minimum length tree
    for j in range(areas):
        
        print 'starting scanning trees for area ' + str(j+1)
        for i in range(trees_number_per_area[j]):
            # opening input file
            points = np.genfromtxt(trees_path + "/Clusters Area %i/cluster_%i.csv" % (j+1, i+1),
                           delimiter=",", skip_header=True)[:, 0:6]
            if len(points) < minimum:
                minimum = len(points)
            # printing progress on screen
            if (i % 100) == 0:
                print 'processed ' + str(i+1) + '/' + str(trees_number_per_area[j]) + ' trees in area ' + str(j+1)

    # print minimum
    print "minimum size tree is " + str(minimum)    
    
    # returning minimum
    return minimum
    
'''
### USAGE ###

This functions trains the stacked denoising autoencoder neural network

Input:
	- input_size: number of trees to consider as input
	- hidden_size: size of the hidden layers expressed as list
	  (i.e if you want 3 hidden layers of sizes 10, 20, 30 you type [10,20,30])
	- number_of_features: number of features to be extracted by the nerual network. This will become the last hidden layer.
	  (i.e. if in the previous step you chose the 3 hidden layers and now you choose 40, overall you'll have [10, 20, 30, 40])
	- training_epochs: number of epochs used to train the nerual network
	- areas: number of areas in which the point cloud surface is divided or that you want to consider
	- trees_path: path of the folder containing the segmented trees
    - trees_locations: path to the csv containing the ground truth trees locations(in UTM, see sample file in Data)
    - area_ids: list containing the known ids of trees with ground truth and location in the area.
                An example for the four areas I tested is the following:
                area_ids = [list(set(range(1, 14)) & valid_ids), list(set(range(14, 35)) & valid_ids),
                            list(set(range(36, 103)) & valid_ids), list(set(range(151, 220)) & valid_ids)]
                area_ids[2].extend([220, 221])
    - minimum: number of points for the smallest tree in the folder
Output:
    - the functions will return the traine stacked denoising autoencoder neural network
    - the function will returnt the max_min list which will be used to normalize test data
'''
def trainNetwork(input_size, hidden_sizes, number_of_features, training_epochs, areas, trees_path,
                 trees_locations, area_ids, minimum):
    
	# corruption levels for the network
	corruption_levels = [.1, .2, .3, .33, .36, .39]
	# getting the trees location from the csv
	locations = pd.DataFrame.from_csv(trees_locations, header=0, index_col=None)
	# adding a dumb column to mark locations
	locations['marked'] = 0

	# setting number of trees in each area
	trees_number_per_area = []
	for i in range(areas):
		trees_number_per_area.append(len([name for name in os.listdir(trees_path + "/Clusters Area %i" % (i+1))
                                           if name.endswith(".csv")]))


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

	# adding values to the train set
	for j in range(areas):
		for i in range(trees_number_per_area[j]):
			# opening input file
			points = np.genfromtxt(trees_path + "/Clusters Area %i/cluster_%i.csv" % (j+1, i+1),
								   delimiter=",", skip_header=True)[:, 0:6]
			locations_area = locations.ix[locations['id'].isin([id for id in locations['id'] if id in area_ids[j]])]
			# checking if the tree is associated to a known location
			tree_id = is_in_locations(locations_area, points)
			# checking if a tree has been found and in case of positive outcome skip it for the training phase
			if not tree_id:
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
				input_size -= 1
				if input_size == 0:
					break
        # if collected enough trees break        
		if input_size == 0:
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

	# training stacked autoencoders and returning trained network and max list
	return train_SdA(train=train, minimum=minimum, hidden_sizes=hidden_sizes,
				    corruption_levels=corruption_levels, n_features=number_of_features,
                      pretraining_epochs=training_epochs), max_min



'''
### USAGE ###
	
 Functions to extract features from the trees point structure in an automatic way 
 by means of an stacked denoising autoencoder neural network

Input:
	- out_path: path of the folder where to store the csv containing features and characteristics
	- areas: number of areas in which the point cloud surface is divided
	- trees_path: path of the folder containing the segmented trees
	- trees_labels: path of the csv containing the trees labels
    - trees_locations: path to the csv containing the ground truth trees locations(in UTM, see sample file in Data)
    - area_ids: list containing the known ids of trees with ground truth and location in the area.
                An example for the four areas I tested is the following:
                area_ids = [list(set(range(1, 14)) & valid_ids), list(set(range(14, 35)) & valid_ids),
                            list(set(range(36, 103)) & valid_ids), list(set(range(151, 220)) & valid_ids)]
                area_ids[2].extend([220, 221])
    - minimum: number of points for the smallest tree in the folder
    - sda: trained instance of a stacked denoising autoencoder
    - max_min: list containing the maximu for all features which is used to normalize test data
Output:
	- the functions will return the traine stacked denoising autoencoder neural network
'''
def extractFeaturesNeuralNetwork(out_path, areas, trees_path, trees_labels,
                                 trees_locations, area_ids, minimum, sda, max_min):
    

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
        # counting trees in specific area
        trees_number_per_area.append(len([name for name in os.listdir(trees_path + "/Clusters Area %i" % (i+1))
                                          if name.endswith(".csv")]))
        # filtering aread ids for valid ones
        area_ids[i] = list(set(area_ids[i]) & valid_ids)    
    	
    # printing progress
    print 'Started retrieving the test data'
    # setting counters to keep track of the number of found trees
    trees_total = 0    
    found_trees_total = 0
    found_trees_areas = [[] for i in range(areas)]
    found_ids = []
    
    # creating empty list for test and labels
    test = []
    # retrieving test data and computing characteristics
    with open(out_path + "/trees_characteristics.csv", 'w') as trees_characteristics:
        # creating writer for tree characteristics
        fieldnames_characteristics = ['height', 'centroid_x', 'centroid_y', 'delta_x', 'delta_y', 'see/notSee']
        writer_characteristics = csv.DictWriter(trees_characteristics, fieldnames=fieldnames_characteristics)
        writer_characteristics.writeheader()
        for j in range(areas):
            print 'starting processing trees for area ' + str(j+1)
            for i in range(trees_number_per_area[j]):
                # saving points in numpy array
                points = np.genfromtxt(trees_path + "/Clusters Area %i/cluster_%i.csv" % (j+1, i+1),
                                       delimiter=",", skip_header=True)[:, 0:6]
                locations_area = locations.ix[locations['id'].isin([id for id in locations['id'] if id in area_ids[j]])]
                # checking if the tree is associated to a known location
                tree_id = is_in_locations(locations_area, points)
                # checking if a tree is corresponding to one of the locations
                if tree_id and tree_id in valid_ids:
                    # if more than minimum number of points then we sample
                    if len(points) > minimum:
                        points = points[np.random.randint(len(points), size=minimum)]
                    # adding points to the test list
                    test.append(points.flatten())
                    # updating list to keep track of found trees per area
                    found_trees_areas[j].append(tree_id)
                    # mark location
                    locations.set_value(locations['id'] == tree_id, 'marked', 1)
                    # adding id to found ids
                    found_ids.append(tree_id)
                    # write computed characteristics on csv
                    writer_characteristics.writerow({'height': height(points),
                                                     'centroid_x': np.mean(points.transpose()[0]),
                                                     'centroid_y': np.mean(points.transpose()[1]),
                                                     'delta_x': delta_x(points),
                                                     'delta_y': delta_y(points),
                                                     'see/notSee': int(labels[labels['id'] == tree_id]['see/notSee'])})    
                # printing progress on screen
                if (i % 100) == 0:
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
    trees_features = open(out_path + "/trees_features_neural_network.csv", 'w')
    # creating csv where to write the computed features and the trees properties
    for i in range(len(result[0])):
        trees_features.write(str(result[0][i].tolist())[1:-1] + ", " +
                             str(int(found_ids[i])))
        trees_features.write('\n')
    trees_features.close()
    
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
    		