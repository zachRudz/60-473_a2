# Used for importing dataset
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from classifier import printMeasuresOfEfficiency
from crossValidation import cross_validate
from plot import plotGrid

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
def fitModel(xTrain, yTrain, n_neighbors=1):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', algorithm='auto',
                               metric=knn_distance_function)
    clf.fit(xTrain, yTrain)
    return clf

def classify(n_neighbors=1, cross_validation=True):
    print("-- K-nearest Neighbors --")
    for ds_file in dataset_files:
        # Isolating features and resulting y value
        dataset = pd.read_csv(ds_file, header=None)
        x = dataset.loc[:, 0:1]
        y = dataset.loc[:, 2]

        # When building our model, should we use cross validation, or just split the data?
        if cross_validation:
            clf, score, xTrain, xTest, yTrain, yTest = cross_validate(fitModel, x, y)

        else:
            # Splitting into test/train sets
            xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=test_size)

            # Fitting our model
            clf = fitModel(xTrain, yTrain, n_neighbors=1)
            score = clf.score(xTest, yTest)

        # Plot twice; Once without color (ie: "unclassified" values), and once with color
        # THis was taking a while on my machine, please uncomment if you're interested
        #plotGrid(clf, x, y, ds_file, 1, colored=False)
        plotGrid(clf, x, y, ds_file, 1)

        # Making predictions on the test set
        y_pred = clf.predict(xTest)
        #print(yTest.shape)
        #print(y_pred.shape)
        #print(confusion_matrix(yTest, y_pred))


        # Printing results
        print("Dataset: {}\tScore: {}".format(ds_file, score))
        printMeasuresOfEfficiency(yTest, y_pred)


def findBestValueOfK():
    print("-- Finding best value of K for K-nearest Neighbors --")
    for ds_file in dataset_files:
        # Isolating features and resulting y value
        dataset = pd.read_csv(ds_file, header=None)
        x = dataset.loc[:, 0:1]
        y = dataset.loc[:, 2]

        # Only searching from values [1..sqrt(numSamples)]
        maxKValue = math.sqrt(len(x[0]))
        maxKValue = math.floor(maxKValue)

        # Recording the score of each value of k, so we can compare to find the best.
        bestKValue = 0
        bestScore = 0

        print("Dataset: {}".format(ds_file))
        for k in range(1,maxKValue):
            # Splitting into test/train sets
            # Not using cross validation because my CV implimentation is broken
            xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=test_size)

            # Fitting our model
            clf = fitModel(xTrain, yTrain, n_neighbors=k)
            score = clf.score(xTest, yTest)

            # Making predictions on the test set
            y_pred = clf.predict(xTest)
            #print(yTest.shape)
            #print(y_pred.shape)
            #print(confusion_matrix(yTest, y_pred))

            # Printing results
            print("\tk: {} Score: {}".format(k, score))

            # Comparing to our best value of k so far
            if(score > bestScore):
                bestKValue = k
                bestScore = score

        # Newline
        print("\tThe best value of k is {} with a score of {}".format(bestKValue, bestScore))
        print()

