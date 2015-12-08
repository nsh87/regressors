#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    'numpy>=1.6.1',
    'scipy>=0.9',
    'matplotlib',
    'scikit-learn>=0.17',
    'statsmodels>=0.6.1',
    'seaborn',
    'pandas',
]

test_requirements = [
    'unittest2',
    'sphinx',
    'sphinx_rtd_theme',
]

setup(
    name='regressors',
    version='0.0.3',
    description="Easy utilities for fitting various regressors, extracting "
                "stats, and making relevant plots",
    long_description=readme + '\n\n' + history,
    author="Nikhil Haas",
    author_email='nikhil@nikhilhaas.com',
    url='https://github.com/nsh87/regressors',
    packages=[
        'regressors',
    ],
    package_dir={'regressors':
                 'regressors'},
    include_package_data=True,
    install_requires=requirements,
    license="ISCL",
    zip_safe=False,
    keywords='regression, lasso, ridge, principal components regression, '
             'elastic net, linear model',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    use_2to3=True
)
