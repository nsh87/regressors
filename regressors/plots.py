# -*- coding: utf-8 -*-

"""This module contains functions for making plots relevant to regressors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn.apionly as sns
import sklearn.decomposition as dcomp
import statsmodels.api as sm

from . import regressors
from . import stats
# TODO: Check these imports
import pandas as pd


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
        Type of residuals to return: 'raw', 'standardized', 'studentized'.
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
        (x-axis, y-axis). Defaults to (10, 8).

    Returns
    -------
    matplotlib.figure.Figure
        The Figure instance.
    """
    # Ensure we only plot residuals using classifiers we have tested
    assert isinstance(clf, regressors.supported_linear_models), (
        "Classifiers of type {0} not currently supported.".format(type(clf)))
    # Get residuals or standardized residuals
    resids = stats.residuals(clf, X, y, r_type)
    predictions = clf.predict(X)
    # Prepare plot labels to use, depending on which type of residuals used
    y_label = {'raw': 'Residuals', 'standardized': 'Standardized Residuals',
               'studentized': 'Studentized Residuals'}
    # Set plot style
    sns.set_style("whitegrid")
    sns.set_context("talk")  # Increase font size on plot
    # Generate residual plot
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


def plot_scree(clf, xlim=[-1, 10], ylim=[-0.1, 1.0], required_var=0.90,
               figsize=(10, 5)):
    """Create side-by-side scree plots for analyzing variance of principle
    components from PCA.

    Parameters
    ----------
    clf : sklearn.decomposition.PCA
        A fitted scikit-learn PCA model.
    xlim : list
        X-axis range. If `required_var` is supplied, the maximum x-axis value
        will automatically be set so that the required variance line is visible
        on the plot. Defaults to [-1, 10].
    ylim : list
        Y-axis range. Defaults to [-0.1, 1.0].
    required_var : float, int, None
        A value of variance to distinguish on the scree plot. Set to None to
        not include on the plot. Defaults to 0.90.
    figsize : tuple
        A tuple indicating the size of the plot to be created, with format
        (x-axis, y-axis). Defaults to (10, 5).

    Returns
    -------
    matplotlib.figure.Figure
        The Figure instance.
    """
    # Ensure we have the a PCA model
    assert isinstance(clf, dcomp.PCA), (
        "Models of type {0} are not supported. Only models of type "
        "sklearn.decomposition.PCA are supported.".format(type(clf))
    )
    # Extract variances from the model
    variances = clf.explained_variance_ratio_
    # Set plot style and scale up font size
    sns.set_style("whitegrid")
    sns.set(font_scale=1.2)
    # Set up figure and generate subplots
    try:
        fig = plt.figure('scree', figsize=figsize)
        # First plot (in subplot)
        plt.subplot(1, 2, 1)
        plt.xlabel("Component Number")
        plt.ylabel("Proportion of Variance Explained")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.plot(variances, marker='o', linestyle='--')
        # Second plot (in subplot)
        cumsum = np.cumsum(variances)  # Cumulative sum of variances explained
        plt.subplot(1, 2, 2)
        plt.xlabel("Number of Components")
        plt.ylabel("Proportion of Variance Explained")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.plot(cumsum, marker='o', linestyle='--')
        # Add marker for required variance line
        if required_var is not None:
            required_var_components = np.argmax(cumsum >= required_var) + 1
            # Update xlim if it is too small to see the marker
            if xlim[1] <= required_var_components:
                plt.xlim([xlim[0], required_var_components + 1])
            # Add the marker and legend to the plot
            plt.axvline(x=required_var_components,
                        c='r',
                        linestyle='dashed',
                        label="> {0:.0f}% Var. Explained: {1} components".format(
                            required_var * 100, required_var_components)
                        )
            legend = plt.legend(loc='lower right', frameon=True)
            legend.get_frame().set_facecolor('#FFFFFF')
        plt.show()
    except:
        raise  # Re-raise the exception
    finally:
        sns.reset_orig()
    return fig


def qq_plot(clf, X, y, figsize=(7, 7)):
    """Generate a Q-Q plot (a.k.a. normal quantile plot).

    Parameters
    ----------
    clf : sklearn.linear_model
        A scikit-learn linear model classifier with a `predict()` method.
    X : numpy.ndarray
        Training data used to fit the classifier.
    y : numpy.ndarray
        Target training values, of shape = [n_samples].
    figsize : tuple
        A tuple indicating the size of the plot to be created, with format
        (x-axis, y-axis). Defaults to (7, 7).

    Returns
    -------
    matplotlib.figure.Figure
        The Figure instance.
    """
    # Ensure we only plot residuals using classifiers we have tested
    assert isinstance(clf, regressors.supported_linear_models), (
        "Classifiers of type {0} not currently supported.".format(type(clf)))
    residuals = stats.residuals(clf, X, y, r_type='raw')
    prob_plot = sm.ProbPlot(residuals, scipy.stats.t, fit=True)
    # Set plot style
    sns.set_style("darkgrid")
    sns.set(font_scale=1.2)
    # Generate plot
    try:
        # Q-Q plot doesn't respond to figure size, so prep a figure first
        fig, ax = plt.subplots(figsize=figsize)
        prob_plot.qqplot(line='45', ax=ax)
        plt.title("Normal Quantile Plot")
        plt.xlabel("Theoretical Standardized Residuals")
        plt.ylabel("Actual Standardized Residuals")
        plt.show()
    except:
        raise  # Re-raise the exception
    finally:
        sns.reset_orig()
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




