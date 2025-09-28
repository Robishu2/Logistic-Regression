import pandas as pd
import numpy as np

# Function to split X and y into a train and test set
def split_data(X, y, test_size=0.3, random_state=12345, include_intercept=True):
    # Raise an error if the dimensions don't match
    if len(X) != len(y):
        raise ValueError('Please ensure that y and X have the same number of rows')

    # Add a new column of ones which will act like an intercept
    if include_intercept:
        intercept = np.ones(len(X))
        intercept = pd.DataFrame({'intercept': intercept})
        X = pd.concat([intercept, X], axis=1)

    # Randomly sample the test set indexes
    idx = y.sample(frac=test_size, random_state=random_state).index

    X_train, X_test = X.drop(idx), X.loc[idx]
    y_train, y_test = y.drop(idx), y.loc[idx]

    return X_train, X_test, y_train, y_test

def make_folds(X, folds=5):
    if folds <= 1:
        raise ValueError('Please ensure that the number of folds >= 2')

    all_idxs = X.index
    all_idxs = np.random.permutation(all_idxs)
    idxs_split = []
    num_per_idxs = len(X)//folds

    for fold in range(folds):
        if (folds > 1) & (fold != folds-1):
            idx = all_idxs[fold*num_per_idxs:(fold+1)*num_per_idxs]
        else:
            idx = all_idxs[fold*num_per_idxs:]

        temp = []
        temp.append(idx)

        not_in_idxs = [s for s in all_idxs if s not in idx]
        temp.append(not_in_idxs)

        idxs_split.append(temp)

    return idxs_split