from __future__ import division


import numpy as np
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

For the parameter 'end' in the manual extracted features case a value equal to 6 means that we
are excluding the RGB features, a value of -1 that we are considering all features and a value for
'start' of 6 and 'end' of 9 that we are using RGB only features

NB: when using the manually extracted features using SVM with LINEAR kernel will result
    in the procedure being stucked, thus use a POLYNOMIAL kernel. When testing with the automacally
    extracted features instead use a LINEAR kernel.

'''

start = 0 # 6 for RGB only
end = -1 # 9 for RGB only

# number of trees for the random forest
ntree = 150
# spatial
spatial = 1.5
# curvature
curvature = 0.2
# list where to store predictions for the statistical significance test
predictions = []
# support lists
n = 10
accuracies = []
recalls = []
precisions = []
# getting the data in csv format
trees_features = pd.DataFrame.from_csv('INSERT_CSV_PATH_HERE' , header=0, index_col=None)

# stratifying the dataset
see = trees_features[trees_features.ix[:, trees_features.columns.size-1] == 1]
see.index = range(1, len(see)+1)
not_see = trees_features[trees_features.ix[:, trees_features.columns.size-1] == 0]
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
    # performing prediction various times
    for j in range(n):
        # initialize list to store prediction
        prediction = []
        # prediction metrics
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        # iterating over the rows
        for i in range(1, len(stratified_trees)+1):
            # creating test and train
            test = stratified_trees.ix[i, :]
            train = stratified_trees.drop(i)
            features = train.columns.size - 2
            label = train.columns.size - 1
            # fitting the classifier
            classifier = classifier.fit(train.ix[:, start:end], train.ix[:, label])
            # predicting test values
            prediction.append(classifier.predict(test[start:end])[0])
            # updating prediction metrics
            if prediction[-1] == test[label] and test[label] == 1:
                tp += 1
            elif prediction[-1] == test[label] and test[label] == 0:
                tn += 1
            elif prediction[-1] != test[label] and test[label] == 0:
                fp += 1
            else:
                fn += 1
        # computing metrics
        accuracies.append((tp+tn)/(tp+tn+fp+fn))
        recalls.append(tp/(tp+tn))
        precisions.append(tp/(tp+fp))
    # printing final progress
    print 'LOOCV using ' + str(classifier).split('(')[0] + ' completed, the resulting metrics are:'
    print 'accuracy is ' + str(np.mean(accuracies)) +  ' +- ' + str(np.std(accuracies))
    print 'recall is ' + str(np.mean(recalls)) + ' +- ' + str(np.std(recalls))
    print 'precision is ' + str(np.mean(precisions)) + ' +- ' + str(np.std(precisions))
    # appending predictions to the predictions list
    predictions.append(prediction)
    # reset lists
    accuracies = []
    recalls = []
    precisions = []
# Computing statistical significance using the last prediction obtained with each classifier
significance = st.f_oneway(stratified_trees.ix[:, stratified_trees.columns.size-1],
                           predictions[0], predictions[1], predictions[2],
                           predictions[3], predictions[4], predictions[5])
print 'The p-value of the Anova test is: ' + str(significance[1])