#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_regressors
---------------

Tests for the `regressors` module.
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import unittest2 as unittest
from sklearn import datasets
import pandas as pd
import numpy as np

from regressors import regressors


class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        boston = datasets.load_boston()
        which_betas = np.ones(13, dtype=bool)
        which_betas[3] = False  # Eliminate dummy variable
        self.X = boston.data[:, which_betas]
        self.y = boston.target

    def tearDown(self):
        pass

    def test_LinearRegression_fit_with_no_xlabels(self):
        ols = regressors.LinearRegression()
        try:
            ols.fit(self.X, self.y)
        except Exception as e:
            self.fail("Fitting with no xlabels raised unexpected "
                      "exception: {0}".format(e))

    def test_LinearRegression_fit_with_xlabels_as_args(self):
        ols = regressors.LinearRegression()
        labels = ['LABEL{0}'.format(i) for i in range(self.X.shape[1])]
        try:
            ols.fit(self.X, self.y, labels)
        except Exception as e:
            self.fail("Fitting with xlabels as *args raised unexpected "
                      "exception: {0}".format(e))

    def test_LinearRegression_fit_with_xlabels_as_kwargs(self):
        ols = regressors.LinearRegression()
        labels = ['LABEL{0}'.format(i) for i in range(self.X.shape[1])]
        try:
            ols.fit(self.X, y=self.y, xlabels=labels)
        except Exception as e:
            self.fail("Fitting with xlabels as **kwargs raised unexpected "
                      "exception: {0}".format(e))

    def test_LinearRegression_fit_with_xlabels_mixed_kwarg(self):
        ols = regressors.LinearRegression()
        labels = ['LABEL{0}'.format(i) for i in range(self.X.shape[1])]
        try:
            ols.fit(self.X, self.y, xlabels=labels)
        except Exception as e:
            self.fail("Fitting with xlabels as **kwargs with y also as "
                      "**kwargs raised unexpected exception: {0}".format(e))

    def test_LinearRegression_fit_with_xlabels_all_kwargs(self):
        ols = regressors.LinearRegression()
        labels = ['LABEL{0}'.format(i) for i in range(self.X.shape[1])]
        try:
            ols.fit(X=self.X, y=self.y, xlabels=labels)
        except Exception as e:
            self.fail("Fitting with xlabels with all parameters as "
                      "**kwargs raised unexpected exception: {0}".format(e))

    def test_LinearRegression_fit_with_xlabels_out_of_position_kwargs(self):
        ols = regressors.LinearRegression()
        labels = ['LABEL{0}'.format(i) for i in range(self.X.shape[1])]
        try:
            ols.fit(X=self.X, xlabels=labels, y=self.y)
        except Exception as e:
            self.fail("Fitting with xlabels with all parameters as "
                      "**kwargs raised unexpected exception: {0}".format(e))

    def test_LinearRegression_fit_with_xlabels_args_out_of_pos_args_fails(self):
        ols = regressors.LinearRegression()
        labels = ['LABEL{0}'.format(i) for i in range(self.X.shape[1])]
        with self.assertRaises(AssertionError):
            ols.fit(self.X, labels, self.y)

    def test_LinearRegression_xlabel_dimensions_error_checking(self):
        ols = regressors.LinearRegression()
        with self.assertRaises(AssertionError):
            ols.fit(self.X, self.y, xlabels=['LABEL1', 'LABEL2'])

    def test_LinearRegression_summary(self):
        ols = regressors.LinearRegression()
        labels = ['LABEL{0}'.format(i) for i in range(self.X.shape[1])]
        ols.fit(self.X, self.y, labels)
        summary = ols.summary()
        self.assertIsInstance(summary, pd.core.frame.DataFrame)
        try:
            str(summary)
        except Exception as e:
            self.fail("str(summary) raised "
                      "exception unexpectedly: {0}".format(e))

if __name__ == '__main__':
    import sys
    sys.exit(unittest.main())
