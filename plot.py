import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Plot the points onto a graph. Rather, onto 2 graphs (one for uniform weights, and another for distance weights).
# Not sure the difference to be honest, but I'd rather keep the example as is just in case.
#
# Set the $colored parameter to False if you want a black and white graph (effectively plotting before classification)
# Source:
# http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
def plotGrid(classifier, x, y, n_neighbors=1, colored=True):
    h = .02  # step size in the mesh

    # Create color maps if the user wishes
    # An uncolored map is essentially an unclassified plot in this case (they're all just ambiguous points on the grid).
    if colored:
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    else:
        cmap_light = ListedColormap([])
        cmap_bold = ListedColormap([])


    # Plotting once for uniform weights, and another for distance
    for weights in ['uniform', 'distance']:
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = x.loc[:, 0].min() - 1, x.loc[:, 0].max() + 1
        y_min, y_max = x.loc[:, 1].min() - 1, x.loc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(x.loc[:, 0], x.loc[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (n_neighbors, weights))