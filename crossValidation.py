from sklearn.model_selection import KFold


def cross_validate(fitModel, x, y, n_splits=10):
    best_score = 0

    # Evaluating the best model via 10-fold cross validation
    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(x):
        xTrain, xTest = x.loc[train_index], x.loc[test_index]
        yTrain, yTest = y.loc[train_index], y.loc[test_index]
        clf = fitModel(xTrain, yTrain)

        # Evaluating the model
        score = clf.score(xTest, yTest)
        if score > best_score:
            best_score = score
            best_xTest = xTest
            best_yTest = yTest
            best_xTrain = xTrain
            best_yTrain = yTrain

    # Training the model based on the best values
    clf = fitModel(xTrain, yTrain)
    #print("best_score: {}  best_xTrain: {}  best_xTest: {}  best_yTrain: {}  best_yTest: {}".format(best_score, best_xTrain, best_xTest, best_yTrain, best_yTest))
    return clf, best_score, best_xTrain, best_xTest, best_yTrain, best_yTest
