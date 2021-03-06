import svm
import datetime


def main():
    startTime = datetime.datetime.now()
    print("Start time: {}".format(startTime))

    # -- Classifier? --
    print("60-473 assignment 02")
    print("0. Quit")
    print("1. Linear")
    print("2. Polynomial")
    print("3. RBF")
    print("4. Calculate ROC curve")
    kernelDecision = input("What kind of SVM kernel do you want to try? ")

    # Parsing decision
    if kernelDecision == "0":
        quit(0)
    elif kernelDecision == "4":
        svm.calculateROC()
        # Printing time analytics
        endTime = datetime.datetime.now()
        diffTime = endTime - startTime
        print("Start time: {}".format(startTime))
        print("End time: {}".format(endTime))
        print("Total elapsed time: {}".format(diffTime))
        return
    elif kernelDecision not in ["1", "2", "3"]:
        print("Not a valid input. Exiting...")
        quit(0)

    # Setting the kernelDecision.g
    if kernelDecision == "1":
        kernel = "linear"
    elif kernelDecision == "2":
        kernel = "poly"
    else:
        kernel = "rbf"


    # -- Cross validation? --
    print("\nUse 10-fold cross validation?")
    print("0. Exit")
    print("1. Yes")
    print("2. No")
    cvDecision = input("What do you want to do? ")

    # Parsing cvDecision
    if cvDecision == "0":
        quit(0)
    elif cvDecision not in ["1", "2"]:
        print("Not a valid input. Exiting...")
        quit(0)
    cross_validation = cvDecision == "1"

    svm.classify(kernel=kernel, cross_validation=cross_validation)

    # Printing time analytics
    endTime = datetime.datetime.now()
    diffTime = endTime - startTime
    print("Start time: {}".format(startTime))
    print("End time: {}".format(endTime))
    print("Total elapsed time: {}".format(diffTime))




if __name__ == "__main__":
    main()
