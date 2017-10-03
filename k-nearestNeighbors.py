# Used for importing dataset
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from plot import plotGrid

################################################################################
#
# Variables
#
################################################################################
test_size = 0.33
knn_distance_function = 'euclidean'  # Equivalent to p=2, according to the documentation
n_neighbors = 1

# Loading dataset
dataset_files = [
    # "datasets/clusterincluster.csv",
    # "datasets/halfkernel.csv",
    # "datasets/twogaussians.csv",
    "datasets/twospirals.csv"
]


################################################################################
#
# Entry Point
#
################################################################################
def main():
    print("-- K-nearest Neighbors --")
    for ds_file in dataset_files:
        # Isolating features and resulting y value
        dataset = pd.read_csv(ds_file, header=None)
        x = dataset.loc[:, 0:1]
        y = dataset.loc[:, 2]

        # Splitting into test/train sets
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=test_size)

        # Fitting our model
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', algorithm='auto',
                                   metric=knn_distance_function)
        clf.fit(xTrain, yTrain)

        # Plot twice; Once without color (ie: "unclassified" values), and once with color
        plotGrid(clf, x, y, 1, colored=False)
        plotGrid(clf, x, y, 1)

        # Evaluating the best model via 10-fold cross validation
        # kf = KFold(n_splits=10)
        # for train, test in kf.split(x):

        score = clf.score(xTest, yTest)

        # Printing results
        print("Dataset: {}\tScore: {}".format(ds_file, score))


if __name__ == "__main__":
    main()
