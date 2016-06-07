from __future__ import division


import pandas as pd
import scipy.stats as st


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


'''
### USAGE ###
This code snippet performs crossvalidation using a csv containing features and class labels for the tree.
In addition, it also performs stratification on the relative dataset.
The tested classifier are:
	- Decision Tree
	- Random Forest
	- Naive Bayes
	- KNN
	- SVM
	- Logistic Regression

For the parameter 'n_features' in the manual extracted features case a value equal to 6 means that we
are excluding the RGB features, a value of -1 that we are considering them all 
	
'''
# number of trees to use for the random forest
ntree = 150
# list where to store predictions for the statistical significance test
predictions = []
# getting the features and labels in csv format
trees_features = pd.DataFrame.from_csv("THE_CSV_PATH_GOES_HERE",
                                       % (spatial, curvature), header=0, index_col=None)
# number of features
n_features = -1

# stratifying the dataset
see = trees_features[trees_features.ix[:, -1] == 1]
see.index = range(1, len(see)+1)
not_see = trees_features[trees_features.ix[:, -1] == 0]
not_see.index = range(1, len(not_see)+1)
bottleneck = min(len(see), len(not_see))
stratified_trees = see.append(not_see.ix[0:bottleneck]) if len(see) == bottleneck\
    else not_see.append(see.ix[0:bottleneck])
stratified_trees.index = range(1, len(stratified_trees)+1)

# dictionary of classifiers
classifiers_dict = {0: KNeighborsClassifier(n_neighbors=3), 1: DecisionTreeClassifier(),
                    2: LogisticRegression(), 3: GaussianNB(), 4: RandomForestClassifier(n_estimators=ntree),
                    5: SVC(kernel='linear')}

# running LOOCV various times with different classifiers
for c in range(len(classifiers_dict)):
    # choosing classifier
    classifier = classifiers_dict[c]
    # initialize list to store prediction
    prediction = []
    # prediction metrics
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(1, len(stratified_trees)+1):
        # creating test and train
        test = stratified_trees.ix[i, :]
        train = stratified_trees.drop(i)
        # fitting the classifier
        classifier = classifier.fit(train.ix[:, 0:n_features], train.ix[:, -1])
        # predicting test values
        prediction.append(classifier.predict(test[0:n_features])[0])
        # updating prediction metrics
        if prediction[-1] == test[-1] and test[-1] == 1:
            tp += 1
        elif prediction[-1] == test[-1] and test[-1] == 0:
            tn += 1
        elif prediction[-1] != test[-1] and test[-1] == 0:
            fp += 1
        else:
            fn += 1
    # appending predictions to the predictions list
    predictions.append(prediction)

    # printing final progress
    print 'LOOCV using ' + str(classifier).split('(')[0] + ' completed, the resulting metrics are:'
    # printing metrics
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    recall = tp/(tp+tn)
    precision = (tp/(tp+fp))
    print 'accuracy is ' + str(accuracy) + ', recall is ' + str(recall) + ', precision is ' + str(precision)

# Computing statistical significance
significance = st.f_oneway(stratified_trees.ix[:, stratified_trees.columns.size-1],
                           predictions[0], predictions[1], predictions[2],
                           predictions[3], predictions[4], predictions[5])
print 'The p-value of the Anova test is: ' + str(significance[1])