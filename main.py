import naiveBayes
import kNearestNeighbors


def main():
    print("60-473 assignment 01")
    print("0. Quit")
    print("1. K-Nearest Neighbors")
    print("2. Naive Bayes")
    decision = input("What classifier do you want to try? ")
    decision = int(decision)

    if decision == 0:
        quit(0)
    elif decision == 1:
        kNearestNeighbors.main()
    elif decision == 2:
        naiveBayes.main()
    else:
        print("Not a valid input. Exiting...")
        quit(0)


if __name__ == "__main__":
    main()
