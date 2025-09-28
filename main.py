import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import time
from logistic_regression import LogisticRegression
from helpers import split_data, tune
from xgboost import XGBClassifier

# Start time
start_time = time.time()

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Create dummies from the categorical columns
df = pd.get_dummies(df)

# Drop the Alive_yes and Alive_no columns for data leakage reasons
df.drop(columns=['alive_yes', 'alive_no'], inplace=True)

# Use SimpleImputer to fill numeric NaNs with column mean
imputer = SimpleImputer(strategy="mean")
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Define target and explanatory variables
y = df['survived']
X = df.drop(columns=['survived']).copy()

# List of potential best number of iterations and learning rate
iterations_list = [1000, 5000, 10000]
learning_rate_list = [0.02, 0.03, 0.04, 0.05]

# Call the tune function to find the best number of iterations and learning rate
best_specifications = tune(X, y, iterations_list, learning_rate_list)
best_iterations = best_specifications['iterations']
best_learning_rate = best_specifications['learning_rate']
print(f'The best specifications of the Logistic Regression model are: {best_specifications}')

# The code below is comparing the Logistic Regression with XGBoost based on a train test split

# Get train and test data
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=3)

# Train the train set
LR = LogisticRegression()
LR.fit_logistic_regression(X_train, y_train, verbose=False, iterations=best_iterations, learning_rate=best_learning_rate)

# Get predictions
y_pred_proba = LR.predict_logistic_regression(X_test)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Print the accuracy
acc = np.mean(y_pred == y_test)
print(f'The accuracy of the XGBoost model was: {acc}')

# Compare to a basic XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Print the accuracy
acc_xgb = np.mean(pred == y_test)
print(f'The accuracy of the XGBoost model was: {acc_xgb}')

# End time
end_time = time.time()
elapsed = end_time - start_time
print(f"The time it took to run the code was {elapsed:.4f} seconds")