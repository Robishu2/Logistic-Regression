import numpy as np

class LogisticRegression:
    # Initialize the betas that will determine the final model after training
    def __init__(self):
        self.betas = None

    # Calculate the betas of the train set
    def fit_logistic_regression(self, X, y, iterations=1000, learning_rate=0.01, verbose=False, early_stopping=float('-inf')):
        # Initialize the betas as zero
        self.betas = np.zeros(len(X.columns))

        for i in range(1, iterations+1):
            # Prints the iteration number, so user can keep track of what stage the model is
            if verbose:
                print(f'We are on iteration {i} of the {iterations} iterations')

            # Use the helper functions to determine the gradient of the iteration
            Z_i = self.Z(X)
            p = self.sigmoid(Z_i)
            grad = self.gradient(X, y, p)

            # Stop training if the improvement is smaller than the early stopping tolerance
            if np.linalg.norm(grad) < early_stopping:
                print('Early stopping is reached!')
                return

            # Calculate the new betas
            self.betas = self.betas + learning_rate*grad

    # Function to calculate X@betas
    def Z(self, X):
        return np.matmul(X, self.betas).to_numpy()

    # Sigmoid function
    def sigmoid(self, Z_i):
        Z = np.zeros(len(Z_i))

        # For loop to calculate sigmoid(z) for each element in Z_i
        for i in range(len(Z)):
            if Z_i[i] >= 0:
                Z[i] = 1/(1+np.exp(-Z_i[i]))
            else:
                Z[i] = np.exp(Z_i[i])/(1+np.exp(Z_i[i]))

        return Z

    # Gradient function
    def gradient(self, X, y, p):
        return np.matmul(X.T, (y-p))

    # Get probabilities based on the betas calculated during training and a new X_test set
    def predict_logistic_regression(self, X):
        Z_i = self.Z(X)
        p = self.sigmoid(Z_i)

        return p