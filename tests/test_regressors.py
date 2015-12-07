#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_regressors
---------------

Tests for the `regressors` module.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import unittest2 as unittest
from sklearn import datasets
from sklearn import metrics
from sklearn import linear_model
from sklearn.decomposition import PCA

from regressors import regressors
from regressors import stats

boston = datasets.load_boston()
which_betas = np.ones(13, dtype=bool)
which_betas[3] = False  # Eliminate dummy variable
X = boston.data[:, which_betas]
y = boston.target


class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_LinearRegression_fit_with_no_xlabels(self):
        ols = regressors.LinearRegression()
        try:
            ols.fit(X, y)
        except Exception as e:
            self.fail("Fitting with no xlabels raised unexpected "
                      "exception: {0}".format(e))

    def test_LinearRegression_fit_with_xlabels_as_args(self):
        ols = regressors.LinearRegression()
        labels = ['LABEL{0}'.format(i) for i in range(X.shape[1])]
        try:
            ols.fit(X, y, labels)
        except Exception as e:
            self.fail("Fitting with xlabels as *args raised unexpected "
                      "exception: {0}".format(e))

    def test_LinearRegression_fit_with_xlabels_as_kwargs(self):
        ols = regressors.LinearRegression()
        labels = ['LABEL{0}'.format(i) for i in range(X.shape[1])]
        try:
            ols.fit(X, y=y, xlabels=labels)
        except Exception as e:
            self.fail("Fitting with xlabels as **kwargs raised unexpected "
                      "exception: {0}".format(e))

    def test_LinearRegression_fit_with_xlabels_mixed_kwarg(self):
        ols = regressors.LinearRegression()
        labels = ['LABEL{0}'.format(i) for i in range(X.shape[1])]
        try:
            ols.fit(X, y, xlabels=labels)
        except Exception as e:
            self.fail("Fitting with xlabels as **kwargs with y also as "
                      "**kwargs raised unexpected exception: {0}".format(e))

    def test_LinearRegression_fit_with_xlabels_all_kwargs(self):
        ols = regressors.LinearRegression()
        labels = ['LABEL{0}'.format(i) for i in range(X.shape[1])]
        try:
            ols.fit(X=X, y=y, xlabels=labels)
        except Exception as e:
            self.fail("Fitting with xlabels with all parameters as "
                      "**kwargs raised unexpected exception: {0}".format(e))

    def test_LinearRegression_fit_with_xlabels_out_of_position_kwargs(self):
        ols = regressors.LinearRegression()
        labels = ['LABEL{0}'.format(i) for i in range(X.shape[1])]
        try:
            ols.fit(X=X, xlabels=labels, y=y)
        except Exception as e:
            self.fail("Fitting with xlabels with all parameters as "
                      "**kwargs raised unexpected exception: {0}".format(e))

    def test_LinearRegression_fit_with_xlabels_args_out_of_pos_args_fails(self):
        ols = regressors.LinearRegression()
        labels = ['LABEL{0}'.format(i) for i in range(X.shape[1])]
        with self.assertRaises(AssertionError):
            ols.fit(X, labels, y)

    def test_LinearRegression_xlabel_dimensions_error_checking(self):
        ols = regressors.LinearRegression()
        with self.assertRaises(AssertionError):
            ols.fit(X, y, xlabels=['LABEL1', 'LABEL2'])

    def test_LinearRegression_summary(self):
        ols = regressors.LinearRegression()
        labels = ['LABEL{0}'.format(i) for i in range(X.shape[1])]
        ols.fit(X, y, labels)
        summary = ols.summary()
        self.assertIsInstance(summary, pd.core.frame.DataFrame)
        try:
            str(summary)
        except Exception as e:
            self.fail("str(summary) raised "
                      "exception unexpectedly: {0}".format(e))


class TestStatsResiduals(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_classifier_type_assertion_raised(self):
        # Test that assertion is raised for unsupported model
        pcomp = PCA()
        pcomp.fit(X, y)
        with self.assertRaises(AttributeError):
            stats.residuals(pcomp, X, y)

    def tests_classifier_type_assertion_not_raised(self):
        # Test that assertion is not raise for supported models
        for classifier in regressors.supported_linear_models:
            clf = classifier()
            clf.fit(X, y)
            try:
                stats.residuals(clf, X, y)
            except Exception as e:
                self.fail("Testing supported linear models in residuals "
                          "function failed unexpectedly: {0}".format(e))

    def test_getting_raw_residuals(self):
        ols = regressors.LinearRegression()
        ols.fit(X, y)
        try:
            stats.residuals(ols, X, y, r_type='raw')
        except Exception as e:
            self.fail("Testing raw residuals failed unexpectedly: "
                      "{0}".format(e))

    def test_getting_standardized_residuals(self):
        ols = regressors.LinearRegression()
        ols.fit(X, y)
        try:
            stats.residuals(ols, X, y, r_type='standardized')
        except Exception as e:
            self.fail("Testing standardized residuals failed unexpectedly: "
                      "{0}".format(e))

    def test_getting_studentized_residuals(self):
        ols = regressors.LinearRegression()
        ols.fit(X, y)
        try:
            stats.residuals(ols, X, y, r_type='studentized')
        except Exception as e:
            self.fail("Testing studentized residuals failed unexpectedly: "
                      "{0}".format(e))

    def test_error_not_raised_by_sse(self):
        # Test that assertion is not raise for supported models
        for classifier in regressors.supported_linear_models:
            clf = classifier()
            clf.fit(X, y)
            try:
                sse = stats.sse(clf, X, y)
            except Exception as e:
                self.fail("Testing SSE function for supported linear models "
                          "failed unexpectedly: {0}".format(e))

    def test_error_not_raised_by_adj_r2_score(self):
        # Test that assertion is not raise for supported models
        for classifier in regressors.supported_linear_models:
            clf = classifier()
            clf.fit(X, y)
            try:
                stats.adj_r2_score(clf, X, y)
            except Exception as e:
                self.fail("Testing adjusted R2 function for supported linear "
                          "models failed unexpectedly: {0}".format(e))

    def test_error_not_raised_by_coef_se(self):
        # Test that assertion is not raise for supported models
        for classifier in regressors.supported_linear_models:
            clf = classifier()
            clf.fit(X, y)
            try:
                stats.coef_se(clf, X, y).shape
            except Exception as e:
                self.fail("Testing adjusted R2 function for supported linear "
                          "models failed unexpectedly: {0}".format(e))

    def test_error_not_raised_by_coef_tval(self):
        # Test that assertion is not raise for supported models
        for classifier in regressors.supported_linear_models:
            clf = classifier()
            clf.fit(X, y)
            try:
                stats.coef_tval(clf, X, y).shape
            except Exception as e:
                self.fail("Testing adjusted R2 function for supported linear "
                          "models failed unexpectedly: {0}".format(e))

if __name__ == '__main__':
    import sys
    sys.exit(unittest.main())
