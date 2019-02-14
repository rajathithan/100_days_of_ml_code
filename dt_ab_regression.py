# Decision trees - Each node contributes to a particular decision 
# that contributes to the final output. Leaf nodes represent the output
# values and branches represent the intermediate decisions made.


# AdaBoost - Fit a regressor on the dataset , compute the error , refit
# the regerssor on the same dataset again based on the error estimate,
# until the desired accuracy is achieved.

import numpy as np  
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn import metrics as sm
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 


# Load housing data from dataset
housing_data = datasets.load_boston()

# Input Data - housing_data.data
# Target price - housing_data.target

# To separate input and output and to get random data
x , y = shuffle(housing_data.data, housing_data.target , random_state=7)

#Split 80% of the dataset as training
num_training = int(0.8 * len (x))
print("\nCount of training variables when split with 80%:\n",num_training)

# Training data & Test data
x_train  = x[:num_training]
print("\nInput Training data set:\n",x_train)
y_train = y[:num_training]
print("\nOutput Training data set:\n",y_train)

x_test  = x[num_training:]
print("\nInput Test data set:\n",x_test)
y_test = y[num_training:]
print("\nOutput Test data set:\n",y_test)

# Take a decision tree with depth 4
dt_regressor = DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(x_train, y_train)


# Lets boost decision tree's performance with AdaBoost with 
# estimators as 400 and random_state as 7

ab_regressor = AdaBoostRegressor(dt_regressor,n_estimators=400, random_state=7)
ab_regressor.fit(x_train, y_train)


# Performance of decision tree regressor

y_pred_dt = dt_regressor.predict(x_test)
mse = sm.mean_squared_error(y_test, y_pred_dt)
evs = sm.explained_variance_score(y_test, y_pred_dt)
print("\n#### Decision Tree performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))


# Performance of decision tree regressor with Adaboost

y_pred_dt = ab_regressor.predict(x_test)
mse = sm.mean_squared_error(y_test, y_pred_dt)
evs = sm.explained_variance_score(y_test, y_pred_dt)
print("\n#### Decision Tree performance with Adaboost ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))



def plot_feature_importances (feature_importances, title, feature_names) :
    # Normalize the importance values
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    # Sort the index values and flip them so that they are arranged in decreasing
    # order of importance
    index_sorted = np.flipud(np.argsort(feature_importances))
    # Center the location of the labels on the X-axis (for display purposes only)
    pos = np.arange(index_sorted.shape[0]) + 0.5
    # Plot the bar graph
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted],align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()

# Feature importance

dt_features = dt_regressor.feature_importances_
ab_features = ab_regressor.feature_importances_

# Decision tree regression will give "RM" as important feature.
plot_feature_importances(dt_features,'Decision Tree regressor', housing_data.feature_names)

# Adaboost regression will give "LSTAT" as the important feature. 
plot_feature_importances(ab_features,'AdaBoost regressor', housing_data.feature_names)