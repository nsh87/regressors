# -*- coding: utf-8 -*-

"""This module contains functions for calculating various statistics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import pandas as pd
from sklearn import metrics


def residuals(clf, X, y, r_type='standardized'):
    """Calculate residuals or standardized residuals.

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

    Returns
    -------
    numpy.ndarray
        An array of residuals.
    """
    # Make sure value of parameter 'r_type' is one we recognize
    assert r_type in ('raw', 'standardized', 'studentized'), (
        "Invalid option for 'r_type': {0}".format(r_type))
    y_true = y.view(dtype='float')
    # Use classifier to make predictions
    y_pred = clf.predict(X)
    # Make sure dimensions agree (Numpy still allows subtraction if they don't)
    assert y_true.shape == y_pred.shape, (
        "Dimensions of y_true {0} do not match y_pred {1}".format(y_true.shape,
                                                                  y_pred.shape))
    # Get raw residuals, or standardized or standardized residuals
    resids = y_pred - y_true
    if r_type == 'standardized':
        resids = resids / np.std(resids)
    elif r_type == 'studentized':
        # Prepare a blank array to hold studentized residuals
        studentized_resids = np.zeros(y_true.shape[0], dtype='float')
        # Calcluate hat matrix of X values so you can get leverage scores
        hat_matrix = np.dot(
            np.dot(X, np.linalg.inv(np.dot(np.transpose(X), X))),
            np.transpose(X))
        # For each point, calculate studentized residuals w/ leave-one-out MSE
        for i in range(y_true.shape[0]):
            # Make a mask so you can calculate leave-one-out MSE
            mask = np.ones(y_true.shape[0], dtype='bool')
            mask[i] = 0
            loo_mse = np.average(resids[mask] ** 2, axis=0)  # Leave-one-out MSE
            # Calculate studentized residuals
            studentized_resids[i] = resids[i] / np.sqrt(
                loo_mse * (1 - hat_matrix[i, i]))
        resids = studentized_resids
    return resids


def sse(clf, X, y):
    """Calculate the standard squared error of the model.

    Parameters
    ----------
    clf : sklearn.linear_model
        A scikit-learn linear model classifier with a `predict()` method.
    X : numpy.ndarray
        Training data used to fit the classifier.
    y : numpy.ndarray
        Target training values, of shape = [n_samples].

    Returns
    -------
    float
        The standard squared error of the model.
    """
    y_hat = clf.predict(X)
    sse = np.sum((y_hat - y) ** 2)
    return sse


def adj_r2_score(clf, X, y):
    """Calculate the adjusted R:superscript:`2` of the model.

    Parameters
    ----------
    clf : sklearn.linear_model
        A scikit-learn linear model classifier with a `predict()` method.
    X : numpy.ndarray
        Training data used to fit the classifier.
    y : numpy.ndarray
        Target training values, of shape = [n_samples].

    Returns
    -------
    float
        The adjusted R:superscript:`2` of the model.
    """
    n = X.shape[0]  # Number of observations
    p = X.shape[1]  # Number of features
    r_squared = metrics.r2_score(y, clf.predict(X))
    return 1 - (1 - r_squared) * ((n - 1) / (n - p - 1))


def se_betas(clf, X, y):
    """Calculate standard error for betas coefficients.
    Parameters
    ----------
    clf : sklearn.linear_model
        A scikit-learn linear model classifier with a `predict()` method.
    X : numpy.ndarray
        Training data used to fit the classifier.
    y : numpy.ndarray
        Target training values, of shape = [n_samples].
    Returns
    -------
    numpy.ndarray
    An array of standard errors of betas coefficients.
    """

    X = np.matrix(X)
    n = X.shape[0]
    X1 = np.hstack((np.ones((n, 1)), np.matrix(X)))
    mat_se = sc.linalg.sqrtm(metrics.mean_squared_error(y, clf.predict(X)) *
                             np.linalg.inv(X1.T * X1))
    se = np.diagonal(mat_se)
    return se


def tval_betas(clf, X, y):
    """Calculate t statistic for betas coefficients.
    Parameters
    ----------
    clf : sklearn.linear_model
        A scikit-learn linear model classifier with a `predict()` method.
    X : numpy.ndarray
        Training data used to fit the classifier.
    y : numpy.ndarray
        Target training values, of shape = [n_samples].
    Returns
    -------
    numpy.ndarray
    An array of t statistic values.
    """
    a = np.array(clf.intercept_ / se_betas(clf, X, y)[0])
    b = np.array(clf.coef_ / se_betas(clf, X, y)[1:])
    tval = np.append(a, b)
    return tval


def pval_betas(clf, X, y):
    """Calculate p values for betas coefficients.
    Parameters
    ----------
    clf : sklearn.linear_model
        A scikit-learn linear model classifier with a `predict()` method.
    X : numpy.ndarray
        Training data used to fit the classifier.
    y : numpy.ndarray
        Target training values, of shape = [n_samples].
    Returns
    -------
    numpy.ndarray
    An array of p_values.
    """
    n = X.shape[0]
    t = tval_betas(clf, X, y)
    p = 2 * (1 - sc.stats.t.cdf(abs(t), n - 1))
    return p


def fsat(clf, X, y):
    """Calculate overall F statistic for betas coefficients.
    Parameters
    ----------
    clf : sklearn.linear_model
        A scikit-learn linear model classifier with a `predict()` method.
    X : numpy.ndarray
        Training data used to fit the classifier.
    y : numpy.ndarray
        Target training values, of shape = [n_samples].
    Returns
    -------
    integer
    integer of F statistic value.
    """
    n = X.shape[0]
    p = X.shape[1]
    r2 = clf.score(X, y)
    f = (r2 / p) / ((1 - r2) / (n - p - 1))
    return f


def summary(clf, X, y, Xlabels):
    sse(clf, X, y)
    adj_r2_score(clf, X, y)
    tval_betas(clf, X, y)
    pval_betas(clf, X, y)
    se_betas(clf, X, y)
    metrics.mean_squared_error(y, clf.predict(X))
    fsat(clf, X, y)

    d = pd.DataFrame(index=['intercept'] + list(Xlabels),
                     columns=['estimate', 'std error', 't value', 'p value'])

    d['estimate'] = np.array([clf.intercept_] + list(clf.coef_))
    d['std error'] = se_betas(clf, X, y)
    d['t value'] = tval_betas(clf, X, y)
    d['p value'] = pval_betas(clf, X, y)

    print('R_squared : ' + str(clf.score(X, y)))
    print('Adjusted R_squared : ' + str(adj_r2_score(clf, X, y)))
    print('F stat : ' + str(fsat(clf, X, y)))
    return d
