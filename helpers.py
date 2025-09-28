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