========
Usage
========

Below are some demonstrations of using Regressors in a project. We'll import a
the Boston data set first to demonstrate the functions' usage::

    from sklearn import datasets
    boston = datasets.load_boston()
    which_betas = np.ones(13, dtype=bool)
    which_betas[3] = False  # Eliminate dummy variable
    X = boston.data[:, which_betas]
    y = boston.target

Obtaining Summary Statistics
----------------------------

There are several functions provided that compute various statistics
about some of the regression models in scikit-learn. These functions are:

    1. :py:func:`regressors.stats.sse`
    2. adj_r2_score(clf,X, y)
    3. mse(clf, X, y)
    4. ssr(clf, X, y)
    5. ssto(clf, X, y)
    6. r2_adj_score(clf,X, y)
    7. se_betas(clf, X, y)
    8. tval_betas(clf, X, y)
    9. pval_betas(clf, X, y)
    10. fsat(clf, X, y)
    11. summary(clf, X, y, Xlabels)

An example with is developed below for a better understanding of these
functions::

    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    Xlabels = boston.feature_names

SSE
~~~

To calculate the SSE:

    ols = linear_model.LinearRegression()
    clf = ols.fit(X, y)
    sse(clf, X, y)

output: 11080.2762841

SSR
~~~

To calculate the SSE:

    ols = linear_model.LinearRegression()
    clf = ols.fit(X, y)
    ssr(clf, X, y)

output: 31636.0191309

SSTO
~~~~

To calculate the SSTO:

    ols = linear_model.LinearRegression()
    clf = ols.fit(X, y)
    ssto(clf, X, y)

output: 42716.295415

Adjusted R2
~~~~~~~~~~~

To calculate the adjusted R2:

    ols = linear_model.LinearRegression()
    clf = ols.fit(X, y)
    r2_adj_score(clf,X, y)

output: 0.740607742865

MSE
~~~

To calculate the MSE:

    ols = linear_model.LinearRegression()
    clf = ols.fit(X, y)
    mse(clf, X, y)

output: 22.5208867564

Standard Error of Beta Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To calculate the standard error of betas coefficients:

    ols = linear_model.LinearRegression()
    clf = ols.fit(X, y)
    se_betas(clf, X, y)

output : [  4.93910982e+00   3.17198514e-02   1.07507294e-02   5.61313621e-02
            8.60206463e-01   3.61026491e+00   2.74628671e-01   9.67057645e-03
            1.81300829e-01   6.18752720e-02   1.05924564e-03   8.94695520e-02
            1.13100913e-03   4.23138697e-02]

T-values of Beta Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To calculate the t values betas coefficients:

    ols = linear_model.LinearRegression()
    clf = ols.fit(X, y)
    tval_betas(clf, X, y)

output : [  7.38819436  -3.37865884   4.31554158   0.37163252   3.12548384
            -4.92921132  13.85417062   0.07766463  -8.13983478   4.93985769
            -11.63974227 -10.65684955   8.30454194 -12.41830721]

P-values of Beta Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To calculate the p values of betas coefficients:

    ols = linear_model.LinearRegression()
    clf = ols.fit(X, y)
    pval_betas(clf, X, y)

output : [  6.20392626e-13   7.84699819e-04   1.91636535e-05   7.10322348e-01
            1.87729071e-03   1.12154925e-06   0.00000000e+00   9.38125595e-01
            3.10862447e-15   1.06468914e-06   0.00000000e+00   0.00000000e+00
            8.88178420e-16   0.00000000e+00]

F-statistic
~~~~~~~~~~~

To calculate the F statistic of betas coefficients:

    ols = linear_model.LinearRegression()
    clf = ols.fit(X, y)
    fsat(clf, X, y)

output : 108.057020999

Summary Statistic Table
~~~~~~~~~~~~~~~~~~~~~~~

The summary statistic table calls all the functions above and generate the statitics
in a more appropriate format.

To obtain the summary table:

    ols = linear_model.LinearRegression()
    clf = ols.fit(X, y)
    summary(clf, X, y)

output :

R_squared : 0.740607742865
Adjusted R_squared : 0.740607742865
F stat : 108.057020999

            estimate  std error    t value       p value
intercept  36.491103   4.939110   7.388194  6.203926e-13
CRIM       -0.107171   0.031720  -3.378659  7.846998e-04
ZN          0.046395   0.010751   4.315542  1.916365e-05
INDUS       0.020860   0.056131   0.371633  7.103223e-01
CHAS        2.688561   0.860206   3.125484  1.877291e-03
NOX       -17.795759   3.610265  -4.929211  1.121549e-06
RM          3.804752   0.274629  13.854171  0.000000e+00
AGE         0.000751   0.009671   0.077665  9.381256e-01
DIS        -1.475759   0.181301  -8.139835  3.108624e-15
RAD         0.305655   0.061875   4.939858  1.064689e-06
TAX        -0.012329   0.001059 -11.639742  0.000000e+00
PTRATIO    -0.953464   0.089470 -10.656850  0.000000e+00
B           0.009393   0.001131   8.304542  8.881784e-16
LSTAT      -0.525467   0.042314 -12.418307  0.000000e+00


    #***********************************
    # * Plot Principal Component Pairs *
    #***********************************

    # Example 1
    import numpy as np
    from sklearn.cross_validation import train_test_split
    iris = sns.load_dataset("iris")  # sample data set
    species = np.array(iris['species'].values, dtype=str)  # set the 'species' aside as Y categorical response variable
    X = iris.iloc[:,:4].as_matrix()  # create matrix of X precictor variables

    X_train, X_test, t_train, t_test = train_test_split(X, species,
                                                    train_size=0.8,
                                                    random_state=1)
    plot_pca_pairs(X_train, t_train, 4, 2, 'hist', 'Species')

    # Example 2
    from sklearn import decomposition
    import numpy as np
    iris = sns.load_dataset("iris")
    species = np.array(iris['species'].values, dtype=str)
    X = iris.iloc[:,:4].as_matrix()
    pcomp = decomposition.PCA(n_components=4)
    pcomp.fit(X)

    plot_pca_pairs(clf_pca=pcomp, x_train=X, n_comps=4, y=species)


    #***********************************
    # * Get Beta Coefficients from PCA *
    #***********************************
    import statsmodels.api as sm
    dta = sm.datasets.fair.load_pandas().data  # sample dataset
    dta['affair'] = (dta['affairs'] > 0).astype(float)  # adds Y to dataframe based on 'affairs' values
    X = dta.ix[:, 0:8].as_matrix()  # want only X data; take Y out; convert it from pandas.dataframe to numpy.matrix
    Y = np.array(dta['affair'])  # set the Y response to a numpy.array

    # perform PCA/PCR. The pcr() function returns a tuple(mspe.mean(), mse.mean(), ols, pcomp)
    tmp = pcr(X, Y, num_components=4, k=10)
    ols = tmp[2]
    pcomp = tmp[3]

    # send the OLS and PCA object into the pca_beta_coeffs() fxn
    print(pca_beta_coeffs(ols, pcomp))

