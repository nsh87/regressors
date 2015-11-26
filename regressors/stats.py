# -*- coding: utf-8 -*-

"""This module contains functions for calculating various statistics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from sklearn import linear_model as lm
import seaborn.apionly as sns
import numpy as np
import matplotlib._pyplot as plt

supported_linear_models = (lm.LinearRegression, lm.Lasso, lm.Ridge,
                           lm.ElasticNet)
# assert isinstance(clf, supported_linear_models), (
#     "Classifiers of type {} not currently supported.".format(type(clf))
# )


def residuals(y_true, y_pred, standardized=True):
    # Make sure dimensions agree (Numpy still allows subtraction if they don't)
    assert y_true.shape == y_pred.shape, (
        "Dimensions of y_true {0} do not match y_pred {1}".format(y_true.shape,
                                                                  y_pred.shape)
    )
    # With sns, only use their API so you don't change user stuff
    sns.set_context("talk")  # Increase font size on plot
    # Get residuals or standardized residuals
    resids = y_pred - y_true  # Residuals
    if standardized is True:
        resids = resids / np.std(resids)  # Standardize resids
    fig = plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    plt.scatter(predictions, standardized_resids, s=14, c='gray',
                alpha=0.7)
    plt.hlines(y=0, xmin=predictions.min() - 100,
               xmax=predictions.max() + 100,
              linestyle='dotted')
    plt.title("Residual Plot")
    plt.xlabel("Predictions")
    plt.ylabel("Standardized Residuals")


