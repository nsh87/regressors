# -*- coding: utf-8 -*-

"""This module contains core classes for regression models."""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from sklearn import linear_model
import numpy as np
import pandas as pd


class LinearRegression(linear_model.LinearRegression):

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=1):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.xlabels = None

    def fit(self, *args, **kwargs):
        """Hijacks :py:class:`skleanr.linear_model.LinearRegression` to accept
        an extra parameter _xlabels_.

        Parameters
        ----------
        xlabels : list, tuple, numpy.ndarray
            Optional labels for the coefficients, corresponding to columns of X.
        """
        # Try to get xlabels from *args or **kwargs
        xlabels = kwargs.get('xlabels')  # Get xlabels or None if not in kwargs
        if xlabels is None:
            try:
                xlabels = args[2]
                args = list(args)
                args.pop()  # Remove xlabels from args
            except IndexError:
                # No xlabels in args, either
                pass
        else:
            del kwargs['xlabels']  # Remove xlabels from kwargs
        # Get X matrix
        try:
            X = kwargs['X']
        except KeyError:
            X = args[0]
        # Make xlabels if none were passed
        ncols = X.shape[1]
        if xlabels is None:
            xlabels = np.array(
                ['x{0}'.format(i) for i in range(ncols)], dtype='str')
        elif isinstance(xlabels, (tuple, list)):
            xlabels = np.array(xlabels, dtype='str')
        # Make sure dims of xlabels matches dims of X
        if xlabels.shape[0] != ncols:
            raise AssertionError(
                "Dimension of xlabels {0} does not match "
                "X {1}.".format(xlabels.shape, X.shape))
        self.xlabels = xlabels
        # Call parent .fit() method
        super(LinearRegression, self).fit(*args, **kwargs)


    def summary(self):
        """Return summary results of fitted model."""
        coeffs = np.concatenate((np.array([self.intercept_]), self.coef_))
        labels = np.concatenate((np.array(['_intercept']), self.xlabels))
        summary = {'coef': pd.Series(coeffs, index=labels)}
        df = pd.DataFrame(summary)
        return df
