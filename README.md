##  Project Strcture

In this file I briefly describe what each of the project files contain:
* FilterPoints.py : this file contains the functions to perform noise removal and compute height   of points w.r.t. the ground points in the point cloud(it requires that ground points in the       point cloud have been already identified)
* TreesSegmentation.py : this file contains the vegetation segmentation algorithm
* ExtractFeatures: contains the function used to manually extract features from the trees point     structure
* ExtractFeaturesNeuralNetwork.py : contains the function using neural networks to extract         features in an automatic way from the trees point structure
* ExtractLocations.py: contains the fucntion used to extract the location in UTM from a KML file   containing coordinates expressed in lat-long
* TreesCrossvalidation.py : this file performs a crossvalidation using the features extracted in   previous steps and the ground truth available
* ComputesLinesOfSight.py : contains the functions used to analyze the lines of sight in the       environment
* Features.py : contains the functions used to extract the features
* UefulFunctions.py : contains a set of auxiliary functions used towards the project
* ComputeLinesOfSight.py : contains the algorithm used to perform lines of sight analysis
* VisibilityNetwork.py : contains the algorithm used to build the visibility network for each timestamp
* The files da.py, mlp.py, logistic_sgd.py and utils.py are taken from the section of deep         learning available on theano website (http://deeplearning.net/tutorial/)
* sda_mod.py: this file contains the modified neural network used to extract features
    
In each file the parameters required in input and the output are described in details
    
## Ground/Non-Ground Identification
    
To identify ground points in the LAS files I used the software mcc-lidar, available at https://sourceforge.net/p/mcclidar/wiki/Home/
