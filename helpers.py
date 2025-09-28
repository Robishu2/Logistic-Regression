import pandas as pd
import numpy as np
from logistic_regression import LogisticRegression

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

    # Select the train and test sets
    X_train, X_test = X.drop(idx), X.loc[idx]
    y_train, y_test = y.drop(idx), y.loc[idx]

    return X_train, X_test, y_train, y_test

# Function to create k folds for cross-validation
def make_folds(X, folds=5):
    # Raise an error if the number of folds is invalid
    if folds <= 1:
        raise ValueError('Please ensure that the number of folds >= 2')

    # Randomly shuffle the indexes
    all_idxs = X.index
    all_idxs = np.random.permutation(all_idxs)

    idxs_split = []
    num_per_idxs = len(X)//folds

    for fold in range(folds):
        # Select the test fold
        if (folds > 1) & (fold != folds-1):
            idx = all_idxs[fold*num_per_idxs:(fold+1)*num_per_idxs]
        else:
            # last fold takes the rest
            idx = all_idxs[fold*num_per_idxs:]

        # Build the [test, train] split
        temp = []
        temp.append(idx)
        not_in_idxs = [s for s in all_idxs if s not in idx]
        temp.append(not_in_idxs)

        idxs_split.append(temp)

    return idxs_split

# Function to tune hyperparameters using cross-validation
def tune(X, y, iter_list, learning_rate_list):
    best_acc = 0
    best_specs = {}

    # Function to tune hyperparameters using cross-validation
    for iterations in iter_list:
        for learning_rate in learning_rate_list:
            splits = make_folds(X)

            accs = []

            # Loop over each fold
            for test_idx, train_idx in splits:
                # Create train and test sets
                X_train, X_test = X.loc[train_idx], X.loc[test_idx]
                y_train, y_test = y.loc[train_idx], y.loc[test_idx]

                # Train logistic regression
                LR = LogisticRegression()
                LR.fit_logistic_regression(X_train, y_train, verbose=False)

                # Get predictions
                y_pred_proba = LR.predict_logistic_regression(X_test)
                y_pred = (y_pred_proba >= 0.5).astype(int)

                # Compute accuracy
                acc = np.mean(y_pred == y_test)
                accs.append(acc)

            # Update best hyperparameters if accuracy improved
            if np.mean(accs) > best_acc:
                best_acc = np.round(np.mean(accs), 3)
                best_specs = {'learning_rate': learning_rate, 'iterations': iterations, 'accuracy': best_acc}

    return best_specs




