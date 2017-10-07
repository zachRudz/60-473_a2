# Used for importing dataset
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from crossValidation import cross_validate
from classifier import printMeasuresOfEfficiency

################################################################################
#
# Variables
#
################################################################################
from plot import plotGrid

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
            clf, score, xTrain, xTest, yTrain, yTest = cross_validate(fitModel, x, y)

        else:
            # No cross validation
            # Splitting into test/train sets
            xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=test_size)
            clf = fitModel(xTrain, yTrain)

            # Printing results
            score = clf.score(xTest, yTest)


        # Plot twice; Once without color (ie: "unclassified" values), and once with color
        # plotGrid(clf, x, y, ds_file, 1, colored=False)
        # plotGrid(clf, x, y, ds_file, 1)

        # Making predictions on the test set
        y_pred = clf.predict(xTest)
        #print(yTest.shape)
        #print(y_pred.shape)
        #print(confusion_matrix(yTest, y_pred))


        # Printing results
        print("Dataset: {}\tScore: {}".format(ds_file, score))
        printMeasuresOfEfficiency(yTest, y_pred)
