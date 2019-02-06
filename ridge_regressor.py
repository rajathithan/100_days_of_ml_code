# Linear regression uses ordinary least squares, which tries to minimize 
# the squares of errors. The outliers tend to cause problems because they 
# contribute a lot to the overall error, To avoid this, we use regularization 
# where a penalty is imposed on the size of the coefficients. This method is 
# called Ridge Regression

import sys
import numpy as np 
from sklearn import linear_model
from sklearn import metrics as sm
from matplotlib import pyplot as plt

filename = sys.argv[1]
x = []
y = []
with open (filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        x.append(xt)
        y.append(yt)

print("\nInput variables:",x)
print("\nOutput variables:",y)

#Split 80% of the dataset as training
num_training = int(0.8 * len (x))
print("\nCount of training variables when split with 80%:\n",num_training)

#Remaining 20% of the dataset as testing
num_test = len(x) - num_training
print("\nCount of test variables when split with 20%:\n",num_test)

# Training data
x_train  = np.array(x[:num_training]).reshape((num_training,1))
print("\nInput Training data set:\n",x_train)
y_train = np.array(y[:num_training])
print("\nOutput Training data set:\n",y_train)

#Test data
x_test = np.array(x[num_training:]).reshape((num_test,1))
print("\nInput Testing data set:\n",x_test)
y_test = np.array(y[num_training:])
print("\nOutput Testing data set:\n",y_test)

# Create ridge regression object
ridge_regressor = linear_model.Ridge(alpha=1.00, fit_intercept=True, max_iter=10000)

# Train the model using the training sets
ridge_regressor.fit(x_train, y_train)

# Predict the training 
y_train_pred = ridge_regressor.predict(x_train)
plt.figure()
plt.scatter(x_train, y_train, color='green')
plt.plot(x_train, y_train_pred, color='black', linewidth=4)
plt.title('Ridge regression - Training data')
plt.show()

# Predict the test 
y_test_pred = ridge_regressor.predict(x_test)
plt.scatter(x_test, y_test, color='green')
plt.plot(x_test, y_test_pred, color='black', linewidth=4)
plt.title('Ridge regression - Test data')
plt.show()

# Mean absolute error - mean absolute error (MAE) is a measure of difference between 
# two continuous variables. Assume X and Y are variables of paired observations that 
# express the same phenomenon. Examples of Y versus X include comparisons of predicted 
# versus observed, subsequent time versus initial time, and one technique of measurement 
# versus an alternative technique of measurement.
print("\nMean absolute error =", round(sm.mean_absolute_error
(y_test, y_test_pred), 2))

# Mean squared error - The MSE assesses the quality of a predictor (i.e., a function 
# mapping arbitrary inputs to a sample of values of some random variable), or an 
# estimator (i.e., a mathematical function mapping a sample of data to an estimate of a 
# parameter of the population from which the data is sampled). It is always non-negative, 
# and values closer to zero are better
print("\nMean squared error =", round(sm.mean_squared_error
(y_test, y_test_pred), 2))

# Median absolute error - This is the median of all the errors in the given dataset. The
# main advantage of this metric is that it's robust to outliers. A single bad point in the
# test dataset wouldn't skew the entire error metric, as opposed to a mean error metric.
print("\nMedian absolute error =", round(sm.median_absolute_error
(y_test, y_test_pred), 2))

# Explained variance score - This score measures how well our model can account for
# the variation in our dataset. A score of 1.0 indicates that our model is perfect
print("\nExplained variance score =", round(sm.explained_variance_score
(y_test, y_test_pred), 2))

# R2 Score - This is pronounced as R-squared, and this score refers to the coefficient of
# determination. This tells us how well the unknown samples will be predicted by our
# model. The best possible score is 1.0, and the values can be negative as well.
print("\nR2 score =", round(sm.r2_score
(y_test, y_test_pred), 2))

