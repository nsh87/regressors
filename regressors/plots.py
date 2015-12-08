# -*- coding: utf-8 -*-

"""This module contains functions for making plots relevant to regressors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn.apionly as sns
import statsmodels.api as sm
from sklearn import decomposition

from . import _utils
from . import stats


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
        * 'standardized' will return the standardized residuals, also known as
          internally studentized residuals, which is calculated as the residuals
          divided by the square root of MSE (or the STD of the residuals).
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
    assert isinstance(clf, _utils.supported_linear_models), (
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
        plt.hlines(y=0, xmin=predictions.min(), xmax=predictions.max(),
                   linestyle='dotted')
        plt.title("Residuals Plot")
        plt.xlabel("Predictions")
        plt.ylabel(y_label[r_type])
        plt.show()
    except:
        raise  # Re-raise the exception
    finally:
        sns.reset_orig()  # Always reset back to default matplotlib styles
    return fig


def plot_scree(clf_pca, xlim=[-1, 10], ylim=[-0.1, 1.0], required_var=0.90,
               figsize=(10, 5)):
    """Create side-by-side scree plots for analyzing variance of principal
    components from PCA.

    Parameters
    ----------
    clf_pca : sklearn.decomposition.PCA
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
    assert isinstance(clf_pca, decomposition.PCA), (
        "Models of type {0} are not supported. Only models of type "
        "sklearn.decomposition.PCA are supported.".format(type(clf_pca)))
    # Extract variances from the model
    variances = clf_pca.explained_variance_ratio_
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
            plt.axvline(x=required_var_components, c='r', linestyle='dashed',
                        label="> {0:.0f}% Var. Explained: {1} "
                              "components".format(required_var * 100,
                            required_var_components))
            legend = plt.legend(loc='lower right', frameon=True)
            legend.get_frame().set_facecolor('#FFFFFF')
        plt.show()
    except:
        raise  # Re-raise the exception
    finally:
        sns.reset_orig()
    return fig


def plot_qq(clf, X, y, figsize=(7, 7)):
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
    assert isinstance(clf, _utils.supported_linear_models), (
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


def plot_pca_pairs(clf_pca, x_train, y=None, n_components=3, diag='kde',
                   cmap=None, figsize=(10, 10)):
    """
    Create pairwise plots of principal components from x data.

    Colors the components according to the `y` values.

    Parameters
    ----------
    clf_pca : sklearn.decomposition.PCA
        A fitted scikit-learn PCA model.
    x_train : numpy.ndarray
        Training data used to fit `clf_pca`, either scaled or un-scaled,
        depending on how `clf_pca` was fit.
    y : numpy.ndarray
        Target training values, of shape = [n_samples].
    n_components: int
        Desired number of principal components to plot. Defaults to 3.
    diag : str
        Type of plot to display on the diagonals. Default is 'kde'.

        * 'kde': density curves
        * 'hist': histograms

    cmap : str
        A string representation of a Seaborn color map. See available maps:
        https://stanford.edu/~mwaskom/software/seaborn/tutorial/color_palettes.
    figsize : tuple
        A tuple indicating the size of the plot to be created, with format
        (x-axis, y-axis). Defaults to (10, 10).

    Returns
    -------
    matplotlib.figure.Figure
        The Figure instance.
    """
    if y is not None:
        assert y.shape[0] == x_train.shape[0], (
            "Dimensions of y {0} do not match dimensions of x_train {1}".format(
                y.shape[0], x_train.shape[0]))
    # Obtain the projections of x_train
    x_projection = clf_pca.transform(x_train)
    # Create a data frame to hold the projections of n_components PCs
    col_names = ["PC{0}".format(i + 1) for i in range(n_components)]
    df = pd.DataFrame(x_projection[:, 0:n_components], columns=col_names)
    # Generate the plot
    cmap = "Greys" if cmap is None else cmap
    color = "#55A969" if y is None else y
    sns.set_style("white", {"axes.linewidth": "0.8", "image.cmap": cmap})
    sns.set_context("notebook")
    try:
        # Create figure instance with subplot and populate the subplot with
        # the scatter matrix. You need to do this so you can access the figure
        # properties later to increase distance between subplots. If you don't,
        # Pandas will create its own figure with a tight layout.
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        from pandas.tools.plotting import scatter_matrix
        axes = scatter_matrix(df, ax=ax, alpha=0.7, figsize=figsize,
                              diagonal=diag, marker='o', c=color,
                              density_kwds={'c': '#6283B9'},
                              hist_kwds={'facecolor': '#5A76A4',
                                         'edgecolor': '#3D3D3D'})
        # Increase space between subplots
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        # Loop through subplots and remove top and right axes
        axes_unwound = np.ravel(axes)
        for i in range(axes_unwound.shape[0]):
            ax = axes_unwound[i]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        plt.show()
    except:
        raise  # Re-raise the exception
    else:
        sns.reset_orig()
        return fig
    finally:
        sns.reset_orig()
