from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


def logistic_regression(clf, X, Y, train_size=0.8, random_state=1):
    # select random train and test set from X and Y
    X_train, X_test, t_train, t_test = train_test_split(X, Y,
                                                        train_size=train_size,
                                                        random_state=random_state)
    # fit the training data to the passed in model
    clf.fit(X_train, t_train)

    # create beta coeff dataframe with intercept if intercept=True
    preds = clf.predict(X_test)
    betas = clf.coef_[0]
    if clf.intercept_ > 0:
        betas = np.append(betas, clf.intercept_[0])

    colnames = X.columns.values
    if clf.intercept_ > 0:
        colnames = np.append(colnames, 'intercept')

    coef_table = pd.DataFrame({'coefficients': betas, '_variables': colnames})

    # calcualte confusion matrix common responses: sensitivity, specificity, precision
    conf_matrix = confusion_matrix(t_test, preds)
    true_pos = conf_matrix[0,0]
    true_neg = conf_matrix[1,1]
    false_pos = conf_matrix[0,1]
    false_neg = conf_matrix[1,0]

    sensitivity = round(true_pos / float((true_pos + false_neg)), 4)
    specificity = round(true_neg / float((false_pos + true_neg)), 4)
    precision = round(true_pos / float((true_pos + false_pos)), 4)

    return clf, coef_table, {"sensitivity": sensitivity}, {"specificity": specificity}, {"precision": precision}



