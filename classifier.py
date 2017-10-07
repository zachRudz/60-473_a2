from sklearn.metrics import confusion_matrix


def printMeasuresOfEfficiency(yTest, y_pred):
    # I was having an issue where the confusion matrix would come out to [[100]] or [[40]].
    # As opposed to how it should be: a 2x2 matrix.
    # After some investigation, I came to the conclusion that the test set only contained one class!
    # Therefore, there is no false positive, no false negative; Only the one class in the test set.
    if(len(confusion_matrix(yTest, y_pred)) == 1):
        print("Test set contains one class only. There is no false positive or false negative; Only the one class.")
        return


    tn, fp, fn, tp = confusion_matrix(yTest, y_pred).ravel()

    # Measures of efficiency
    # ppv: positive predicted values
    # npv: negative predicted values
    # sensitivity (recall): negative predicted values
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    ppv = tp / tp / (tp+fp)
    npv = tn / (tn + fn)

    print("\ttn: {}  fp: {}  fn: {}  tp: {}".format(tn, fp, fn, tp))
    print("\tspecificity: {}".format(specificity, sensitivity, ppv, npv))
    print("\tsensitivity: {}".format(specificity, sensitivity, ppv, npv))
    print("\tppv: {}".format(specificity, sensitivity, ppv, npv))
    print("\tnpv: {}".format(specificity, sensitivity, ppv, npv))
