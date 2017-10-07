import naiveBayes
import kNearestNeighbors


def main():
    # -- Classifier? --
    print("60-473 assignment 01")
    print("0. Quit")
    print("1. K-Nearest Neighbors")
    print("2. Naive Bayes")
    classifier = input("What classifier do you want to try? ")

    # Parsing decision
    if classifier == "0":
        quit(0)
    elif classifier not in ["1", "2"]:
        print("Not a valid input. Exiting...")
        quit(0)

    # -- Cross validation? --
    print("\nUse cross validation?")
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

    # -- Starting classifier --
    if classifier == "0":
        quit(0)
    if classifier == "1":
        kNearestNeighbors.classify(cross_validation=cross_validation)
    elif classifier == "2":
        # -- Naive bayes --
        naiveBayes.classify(cross_validation=cross_validation)


if __name__ == "__main__":
    main()
