# Used for importing dataset
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from plot import plotGrid
from sklearn.model_selection import KFold

################################################################################
#
# Variables
#
################################################################################
test_size = 0.33
knn_distance_function = 'euclidean'  # Equivalent to p=2, according to the documentation
n_neighbors = 1

# The datasets we're classifying.
dataset_files = [
    "datasets/clusterincluster.csv",
    "datasets/halfkernel.csv",
    "datasets/twogaussians.csv",
    "datasets/twospirals.csv"
]


################################################################################
#
# Entry Point
#
################################################################################
def fitModel(xTrain, yTrain):
    # Fitting our model
    clf = GaussianNB()
    clf.fit(xTrain, yTrain)
    return clf

def classify(cross_validation=True):
    print("-- Naive Bayes --")
    for ds_file in dataset_files:
        # Isolating features and resulting y value
        dataset = pd.read_csv(ds_file, header=None)
        x = dataset.loc[:, 0:1]
        y = dataset.loc[:, 2]

        # When building our model, should we use cross validation, or just split the data?
        if cross_validation:
            # Evaluating the best model via 10-fold cross validation
            kf = KFold(n_splits=10)

            for train_index, test_index in kf.split(x):
                print("TRAIN: {} TEST: {}".format(train_index, test_index))
                #xTrain, xTest = x[train_index], x[test_index]
                #yTrain, yTest = y[train_index], y[test_index]
                #clf = fitModel(xTrain, yTrain)
                #score = clf.score(xTest, yTest)

        else:
            # No cross validation
            # Splitting into test/train sets
            xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=test_size)
            clf = fitModel(xTrain, yTrain)
            score = clf.score(xTest, yTest)

        # Making predictions on the test set
        # y_pred = clf.predict(xTest)


        # Plot twice; Once without color (ie: "unclassified" values), and once with color
        # plotGrid(clf, x, y, ds_file, 1, colored=False)
        # plotGrid(clf, x, y, ds_file, 1)

        # Printing results
        print("Dataset: {}\tScore: {}".format(ds_file, score))
