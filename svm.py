# Used for importing dataset
import math
import pandas as pd
from sklearn import svm, metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold

from classifier import printMeasuresOfEfficiency
from crossValidation import cross_validate
from plot import plotGrid
import matplotlib.pyplot as plt

################################################################################
#
# Variables
#
################################################################################
test_size = 0.33
n_folds = 10

# The datasets we're classifying.
dataset_files = [
    "datasets/clusterincluster.csv",
    "datasets/halfkernel.csv",
    "datasets/twogaussians.csv",
    "datasets/twospirals.csv"
]


################################################################################
#
# Helper functions
#
################################################################################
def getMean(intArray):
    total = 0
    for i in intArray:
        total += i

    return total / len(intArray)


################################################################################
#
# Entry Point
#
################################################################################
def classify(kernel, cross_validation=False):
    print("-- SVM --")
    for ds_file in dataset_files:
        # Isolating features and resulting y value
        dataset = pd.read_csv(ds_file, header=None)
        x = dataset.loc[:, 0:1]
        y = dataset.loc[:, 2]

        # Creating our classifier
        clf = svm.SVC(kernel=kernel, gamma=2)

        # When building our model, should we use cross validation, or just split the data?
        if cross_validation:
            best_score = -1

            # Declaring empty int arrays for each measure of efficiency. Used for finding average later on
            tn = [0] * n_folds
            fp = [0] * n_folds
            fn = [0] * n_folds
            tp = [0] * n_folds
            specificity = [0] * n_folds
            sensitivity = [0] * n_folds
            ppv = [0] * n_folds
            npv = [0] * n_folds

            # K-Folding
            kf = KFold(n_splits=n_folds, shuffle=True)
            i = 0
            for index_train, index_test in kf.split(x):
                xTrain, xTest = x.loc[index_train], x.loc[index_test]
                yTrain, yTest = y.loc[index_train], y.loc[index_test]
                clf.fit(xTrain, yTrain)

                # Evaluating the model
                score = clf.score(xTest, yTest)
                if score > best_score:
                    best_score = score
                    best_xTest = xTest
                    best_yTest = yTest
                    best_xTrain = xTrain
                    best_yTrain = yTrain

                # Making predictions on the test fold which had the best training set
                # ie: We found the optimal training set using CV, and we're making predictions of y
                #      using the corresponding test set
                y_pred = clf.predict(x)

                # Fetching the measure of efficiency
                tn[i], fp[i], fn[i], tp[i] = confusion_matrix(y, y_pred).ravel()
                # Measures of efficiency
                # ppv: positive predicted values
                # npv: negative predicted values
                # sensitivity (recall): negative predicted values
                specificity[i] = tn[i] / (tn[i] + fp[i])
                sensitivity[i] = tp[i] / (tp[i] + fn[i])
                ppv[i] = tp[i] / tp[i] / (tp[i] + fp[i])
                npv[i] = tn[i] / (tn[i] + fn[i])
                i = i + 1

            # Now that we've found the best representation of our data via CV,
            # we can fit the classifier on our "best" training set.
            clf.fit(best_xTrain, best_yTrain)

            # Making predictions on the test fold which had the best training set
            # ie: We found the optimal training set using CV, and we're making predictions of y
            #      using the corresponding test set
            #y_pred = clf.predict(xTest)

            # Printing results
            print("Dataset: {}\tScore: {}".format(ds_file, best_score))
            print("Averages of the measures of efficiency are below:")
            print("\ttn: {}  fp: {}  fn: {}  tp: {}".format(getMean(tn), getMean(fp), getMean(fn), getMean(tp)))
            print("\tspecificity: {}".format(getMean(specificity)))
            print("\tsensitivity: {}".format(getMean(sensitivity)))
            print("\tppv: {}".format(getMean(ppv)))
            print("\tnpv: {}".format(getMean(npv)))


        else:
            # Splitting into test/train sets
            xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=test_size)

            # Fitting our model
            clf.fit(xTrain, yTrain)

            # Evaluating the model
            score = clf.score(xTest, yTest)

            # Making predictions on the test set
            y_pred = clf.predict(xTest)

            # Printing results
            print("Dataset: {}\tScore: {}".format(ds_file, score))
            printMeasuresOfEfficiency(yTest, y_pred)

        # Plot the grid
        plotGrid(clf, x, y, ds_file, 1)


# Reference: https://matplotlib.org/users/pyplot_tutorial.html
# Reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
def calculateROC(kernel="rbf"):

    gamma = ['auto', '1', '2', '3', '4']
    dsNum = 0
    for ds_file in dataset_files:
        plt.figure(dsNum + 1)

        # Isolating features and resulting y value
        dataset = pd.read_csv(ds_file, header=None)
        x = dataset.loc[:, 0:1]
        y = dataset.loc[:, 2]

        # Creating our classifier
        clf = svm.SVC(kernel=kernel, degree=2, gamma=2, probability=True)

        # K-Folding
        kf = KFold(n_splits=n_folds, shuffle=True)
        i = 0
        for index_train, index_test in kf.split(x):
            # Train/test indicies
            xTrain, xTest = x.loc[index_train], x.loc[index_test]
            yTrain, yTest = y.loc[index_train], y.loc[index_test]
            clf.fit(xTrain, yTrain)

            # Calculating the ROC curve
            probabilities = clf.predict_proba(xTest)
            fpr, tpr, thresholds = roc_curve(yTest, probabilities[:, 1], pos_label=2)

            # Calculating the area under the curve
            roc_auc = auc(fpr, tpr)

            # Plotting the ROC curve for this fold
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label="ROC fold %d (AUC = %0.2f)" % (i, roc_auc))

            i += 1

        # Plotting the line of luck
        plt.plot([0,1], [0,1], linestyle="--", lw=2, color='r', label="Luck", alpha=.8)

        # Setting up the plot
        plt.xlim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for dataset {}'.format(ds_file))
        plt.legend(loc="lower right")
        plt.show()

        # Moving on to the next dataset/figure
        dsNum += 1

