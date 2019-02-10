# import plot package
from matplotlib import pyplot as plt

# import numpy package
import numpy as np

# import linear regressors
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)

# import mean squared error metric
from sklearn.metrics import mean_squared_error

# import polynomial features
from sklearn.preprocessing import PolynomialFeatures

# import pipeline
from sklearn.pipeline import make_pipeline

# set the random seed at 42
np.random.seed(42)

# Generate the training data set
X = np.random.normal(size=400)
y = np.sin(X)


# Convert X as 2 dimensional nothing but slicing the vector from 1 
# dimension to 2D array, np.newaxis changes 1D to 2D , 2D to 3D and so on
# Here we have converted to row vector to column vector.
X = X[:, np.newaxis]


# Generate the test data set
X_test = np.random.normal(size=200)
y_test = np.sin(X_test)

# Convert X as 2 dimensional
X_test = X_test[:, np.newaxis]

# introduce errors in y_errors dataset to index 0 and in index 3
# assign value 3 to index 0 and 3
y_errors = y.copy()
y_errors[::3] = 3

# introduce errors in X_errors dataset to index 0 and in index 3
# assign value 3 to index 0 and 3
X_errors = X.copy()
X_errors[::3] = 3

# introduce errors in y_errors_large dataset to index 0 and in index 3
# assign value 10 to index 0 and 3
y_errors_large = y.copy()
y_errors_large[::3] = 10

# introduce errors in X_errors_large dataset to index 0 and in index 3
# assign value 10 to index 0 and 3
X_errors_large = X.copy()
X_errors_large[::3] = 10

# Linear regression - linear regression is a linear approach to modelling the 
# relationship between a scalar response (or dependent variable) and one or more 
# explanatory variables (or independent variables).

# Theil sen regression - This estimator can be computed efficiently, and is 
# insensitive to outliers.  It can be significantly more accurate than non-robust
# simple linear regression (least squares) for skewed and heteroskedastic data, and
# competes well against # least squares even for normally distributed data in terms
# of statistical power. It has been called "the most popular nonparametric technique 
# for estimating a linear trend"

# Random sample consensus (RANSAC) is an iterative method to estimate parameters of a 
# mathematical model from a set of observed data that contains outliers, when outliers 
# are to be accorded no influence on the values of the estimates. Therefore, it also can 
# be interpreted as an outlier detection method.It is a non-deterministic algorithm in the 
# sense that it produces a reasonable result only with a certain probability, with this 
# probability increasing as more iterations are allowed.


# Huber regression -  the Huber loss is a loss function used in robust regression, that 
# is less sensitive to outliers in data than the squared error loss.



estimators = [
              ('OLS', LinearRegression()),
              ('Theil-Sen', TheilSenRegressor(random_state=42)),
              ('RANSAC', RANSACRegressor(random_state=42)),
              ('HuberRegressor', HuberRegressor())
              ]
colors = {'OLS': 'turquoise', 'Theil-Sen': 'gold', 'RANSAC': 'lightgreen', 'HuberRegressor': 'black'}
linestyle = {'OLS': '-', 'Theil-Sen': '-.', 'RANSAC': '--', 'HuberRegressor': '--'}
lw = 3

# Returns evenly spaced numbers over a specified interval
x_plot = np.linspace(X.min(), X.max())


# subplot iterator
p = 1

plt.figure(figsize=(15, 10))
for title, this_X, this_y in [
        ('Modeling Errors Only', X_errors, y_errors),
        ('Corrupt X, Small Deviants', X_errors, y),
        ('Corrupt y, Small Deviants', X, y_errors),
        ('Corrupt X, Large Deviants', X_errors_large, y),
        ('Corrupt y, Large Deviants', X, y_errors_large)]:    
    # plot x & y with blue plus signs 
    plt.subplot(3,2,p)   
    plt.plot(this_X[:, 0], this_y, 'b+')     
    p = p + 1     
    for name, estimator in estimators:
        # Construct a pipeline from the given estimators
        # Shorthand for the pipeline constructor
        model = make_pipeline(PolynomialFeatures(3), estimator)
        # fit the model
        model.fit(this_X, this_y)
        # calculate mse
        mse = mean_squared_error(model.predict(X_test), y_test)
        # get y_plot
        y_plot = model.predict(x_plot[:, np.newaxis])
        # plot it                
        plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
                 linewidth=lw, label='%s: error = %.3f' % (name, mse))       
    legend_title = 'Error of Mean\nAbsolute Deviation\nto Non-corrupt Data'
    # place legends on the upper right    
    legend = plt.legend(loc='upper right', frameon=False, title=legend_title,
                        prop=dict(size='x-small'))
    # settings x axis limits
    plt.xlim(-4, 10.2)
    # settings y axis limits
    plt.ylim(-2, 10.2)
    # Set plot title
    plt.title(title)
plt.show()
