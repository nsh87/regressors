# -*- coding: utf-8 -*-

"""This module contains functions for calculating the real-space
coefficients from Principal Component Analysis."""
from __future__ import absolute_import   # do I need to have these?
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pandas as pd
import seaborn as sns
from sklearn import preprocessing, decomposition


def plot_pca_pairs(x, y, n_comps, facet_size=2, diag='kde', legend_title='Response'):
    """Create pairwise plots of principal components from x data.
       Colors the components according to the y values.

    Parameters
    ----------
    x_train : numpy.ndarray
        Training data used to fit the classifier.
    y : numpy.ndarray
        Target training values, of shape = [n_samples].
    n_comps: integer
        Desired number of principal components to plot
    facet_size: integer
        Numerical value representing size (width) of each facet of the pairwise plot
        Default is 2. Units are in inches
    diag: string
        Type of plot to display on the diagonals. Default is 'kde'.

        * 'kde' = density curves
        * 'hist' = histograms
        * 'None' = blank

    legend_title: string
        Allows the user to specify the title of the legend.
        If None is passed, the plot will have no legend and no colors based on y

    Returns
    -------
    Displays a seaborn pairwise plot.
        More fancy plotting options here:
        http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.pairplot.html
    """

    assert len(y) == len(x), (
        "Dimensions of y {0} do not match dimensions of x {1}".format(len(y), len(x)))

    # scale and transform the X data
    scaled_x = preprocessing.StandardScaler().fit(x)
    transformed_x = scaled_x.transform(x)

    # perform PCA
    pcomp = decomposition.PCA(n_components=n_comps)
    pcomp.fit(transformed_x)
    x_projection = pcomp.transform(transformed_x)

    # create Pandas dataframe for pairwise plot of PCA comps
    df = pd.DataFrame(x_projection[:, 0:n_comps], columns=[range(n_comps)])
    df[legend_title] = y  # add Y response variable to PCA dataframe

    # display the plot
    sns.set_style("white")
    sns.pairplot(df, hue=legend_title, diag_kind=diag, size=facet_size)

    # if seaborn doesn't plot for you, try this below. Not as pretty, but it works.
    # import matplotlib.pyplot as plt
    # a = sns.pairplot(df, hue=legend_title, diag_kind=diag, size=facet_size)
    # plt.show(a)

    return



##  uncomment this code and run the Examples

# Example 1
# -------------------
# import numpy as np
# from sklearn.cross_validation import train_test_split
# iris = sns.load_dataset("iris")
# species = np.array(iris['species'].values, dtype=str)
# X = iris.iloc[:,:4].as_matrix()
# #
# X_train, X_test, t_train, t_test = train_test_split(X, species,
#                                                     train_size=0.8,
#                                                     random_state=1)
# plot_pca_pairs(X_train, t_train, 4, 2, 'hist', 'Species')

# Example 2
# --------------------
# import numpy as np
# tcga = pd.read_csv("/Users/alexromriell/Dropbox/School/FallModule2/Machine Learning/MLproject/TCGA_example.txt", sep=" ")
# tumor_types = np.array(tcga['Subtype'].values, dtype=str)
# y = np.array(tcga.iloc[:, 1], dtype='float64')  # 1st (2nd) column; regress on Gene 1
# X = tcga.iloc[:, 2:].as_matrix()
#
# plot_pca_pairs(X, tumor_types, 4, 2, 'kde', 'Tumor Type')