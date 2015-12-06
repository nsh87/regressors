# -*- coding: utf-8 -*-

"""This module contains functions for calculating various statistics and coefficients."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from sklearn.decomposition import PCA
from regressors import supported_linear_models
import numpy as np


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
        Type of residuals to return: ['raw', 'standardized', 'studentized'].
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
        "Invalid option for 'r_type': {0}".format(r_type)
    )
    y_true = y.view(dtype='float')
    # Use classifier to make predictions
    y_pred = clf.predict(X)
    # Make sure dimensions agree (Numpy still allows subtraction if they don't)
    assert y_true.shape == y_pred.shape, (
        "Dimensions of y_true {0} do not match y_pred {1}".format(y_true.shape,
                                                                  y_pred.shape)
    )
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
            np.transpose(X)
        )
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


def pca_beta_coeffs(clf_ols, clf_pca):
    """Calculate the beta coefficients in real-space (instead of PCA-space).

    Parameters
    ----------
    clf_ols : sklearn.linear_model
        A scikit-learn linear model classifier.
    clf_pca : sklearn.decomposition.PCA
        A scikit-learn PCA model.

    Returns
    -------
    np.ndarray
        An array of the real-space beta coefficients from Principal Component Regression
    """
    # Ensure we only calculate coefficients using classifiers we have tested
    assert isinstance(clf_ols, supported_linear_models), (
        "Classifiers of type {0} not currently supported".format(type(clf_ols)))
    assert isinstance(clf_pca, PCA), (
        "Classifiers of type {0} are not supported. "
        "Please use class sklearn.decomposition.PCA.".format(type(clf_pca)))

    return np.dot(clf_ols.coef_, clf_pca.components_)


