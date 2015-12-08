# -*- coding: utf-8 -*-

"""This module contains utilities or variables required for other modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from sklearn import linear_model as lm

supported_linear_models = (lm.LinearRegression, lm.Lasso, lm.Ridge,
                           lm.ElasticNet)

