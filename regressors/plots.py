# -*- coding: utf-8 -*-

"""This module contains functions for making plots relevant to regressors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
import seaborn.apionly as sns
import pandas as pd
from get_pca import pcr
# from regressors.regressors import stats
# from regressors.regressors import supported_linear_models


def plot_residuals(clf, X, y, r_type='standardized', figsize=(10, 8)):
    """Plot residuals of a linear model.
    Parameters
    ----------
    clf : sklearn.linear_model
        A scikit-learn linear model classifier with a `predict()` method.
    X : numpy.ndarray
        Training data used to fit the classifier.
    y : numpy.ndarray
        Target training values, of shape = [n_samples].
    r_type : str
        Type of residuals to return: ['raw', 'standardized', 'studentized'].
        Defaults to 'standardized'.
        * 'raw' will return the raw residuals.
        * 'standardized' will return the Pearson standardized residuals, also
          known as internally studentized residuals, which is calculated as the
          residuals divided by the square root of MSE (or the STD of the
          residuals).
        * 'studentized' will return the externally studentized residuals, which
          is calculated as the raw residuals divided by sqrt(LOO-MSE * (1 -
          leverage_score)).
    figsize : tuple
        A tuple indicating the size of the plot to be created, with format
        (x-axis, y-axis). Defaults to (10, 10).
    Returns
    -------
    matplotlib.figure.Figure
        The Figure instance.
    """
    # Ensure we only plot residuals using classifiers we have tested
    assert isinstance(clf, supported_linear_models), (
        "Classifiers of type {} not currently supported.".format(type(clf)))
    # Set plot style
    sns.set_style("whitegrid")
    sns.set_context("talk")  # Increase font size on plot
    # Get residuals or standardized residuals
    resids = stats.residuals(clf, X, y, r_type)
    predictions = clf.predict(X)
    # Generate residual plot
    y_label = {'raw': 'Residuals', 'standardized': 'Standardized Residuals',
               'studentized': 'Studentized Residuals'}
    try:
        fig = plt.figure('residuals', figsize=figsize)
        plt.scatter(predictions, resids, s=14, c='gray', alpha=0.7)
        plt.hlines(y=0, xmin=predictions.min(),
                   xmax=predictions.max(), linestyle='dotted')
        plt.title("Residuals Plot")
        plt.xlabel("Predictions")
        plt.ylabel(y_label[r_type])
        plt.show()
    except:
        raise  # Re-raise the exception
    finally:
        sns.reset_orig()  # Always reset back to default matplotlib styles
    return fig


def plot_pca_pairs(clf_pca, x_train, n_comps, y=None, scaler=None, facet_size=2, diag='kde', legend_title=None):
    """Create pairwise plots of principal components from x data.
       Colors the components according to the y values.

    Parameters
    ----------
    clf_pca : sklearn.decomposition.PCA
        A scikit-learn PCA model.
    x_train : numpy.ndarray
        Training data used to fit the classifier.
    n_comps: integer
        Desired number of principal components to plot.
    y : numpy.ndarray
        Target training values, of shape = [n_samples].
    scaler : sklearn.preprocessing
        A scaler created to scale the data per the users' pre-specifications.
    facet_size: integer
        Numerical value representing size (width) of each facet of the pairwise plot.
        Default is 2. Units are in inches.
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
    if y is not None:
        assert y.shape[0] == x_train.shape[0], (
            "Dimensions of y {0} do not match dimensions of x {1}".format(y.shape[0],
                                                                          x_train.shape[0])
        )

    # Scale the data if the user passed in a scaler
    if scaler is not None:
        x_train = scaler.transform(x_train)

    # Perform PCA
    x_projection = clf_pca.transform(x_train)

    # create Pandas dataframe for pairwise plot of PCA comps
    # using i+1 due to python's 0 listing.
    col_names = ["PC{0}".format(i+1) for i in range(n_comps)]
    df = pd.DataFrame(x_projection[:, 0:n_comps], columns=col_names)

    # display legend/colors according to user parameters
    if y is not None and legend_title is not None:
        df[legend_title] = y  # add Y response variable to PCA dataframe
    if y is not None and legend_title is None:
        df['Response'] = y
        legend_title = 'Response'

    # display the plot
    sns.set_style("white")
    sns.pairplot(df, hue=legend_title, diag_kind=diag, size=facet_size)

    # if seaborn doesn't plot for you, try this below. Not as pretty, but it works.
    # import matplotlib.pyplot as plt
    # a = sns.pairplot(df, hue=legend_title, diag_kind=diag, size=facet_size)
    # plt.show(a)

    return



