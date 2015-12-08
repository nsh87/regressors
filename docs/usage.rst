========
Usage
========

To use Regressors in a project::

    import regressors

    #************************
    # * Logistic Regression *
    #************************
    import statsmodels.api as sm
    dta = sm.datasets.fair.load_pandas().data  # sample dataset from statsmodels module

    dta['affair'] = (dta['affairs'] > 0).astype(float)  # define 'affair' column as categorical response; 1's and 0's
    X = dta.ix[:, 0:8]  # subset the data to include only variable data
    Y = dta['affair']

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(fit_intercept=False)  # create logistic regression classifier object
    lr.fit(X,Y)  # fit the logistic regression model with X, Y data
    results = logistic_regression(lr, X, Y)
    # results returns a tuple of (lr_class, coefficient table (dataframe), sensitivity, specificity, and precision)


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

