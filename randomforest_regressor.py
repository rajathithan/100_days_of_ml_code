import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics as sm
from sklearn.utils import shuffle
from dt_ab_regression import plot_feature_importances
import matplotlib.pyplot as plt 

def load_dataset(filename):
    file_reader = csv.reader(open(filename, 'r'),delimiter=',')
    x, y = [], []
    for row in file_reader:
        # Since we are forecasting the bicycle demand, the x variables are the 
        # inputs like date, temp, etc and the y variable should the final output
        # the number of people using bicycles

        # x - Take row columns from 2 to 13 - inputs
        # x.append(row[2:13])
        # x.append(row[2:14])
        x.append(row[2:15])
        # y - Take last row column, as it is a summation
        # of 14 & 15th col - outputs
        y.append(row[-1])    
    feature_names = np.array(x[0])
    return np.array(x[1:]).astype(np.float32),np.array(y[1:]).astype(np.float32),feature_names


if __name__ == '__main__':

    filename = 'bike_day.csv'

    # Load the dataset
    x,y,feature_names = load_dataset(filename)

    # shuffle the values
    x,y = shuffle(x,y,random_state=7)

    # Split the dataset for training and testing
    # we will use 90% for training and 10% for testing

    num_training = int(0.9 * len(x))
    x_train, y_train = x[:num_training], y[:num_training]
    x_test, y_test = x[num_training:],y[num_training:]

    print(x_train)
    print(y_train)

    # n_estimators refers to the number of estimators, which is the number of
    # decision trees that we want to use in our random forest. The max_depth parameter
    # refers to the maximum depth of each tree, and the min_samples_split parameter
    # refers to the number of data samples that are needed to split a node in the tree.

    rf_regressor = RandomForestRegressor(n_estimators=1000,max_depth=10,min_samples_split=2)
    rf_regressor.fit(x_train,y_train)

    y_pred = rf_regressor.predict(x_test)

    # Median absolute error - This is the median of all the errors in the given dataset. The
    # main advantage of this metric is that it's robust to outliers. A single bad point in the
    # test dataset wouldn't skew the entire error metric, as opposed to a mean error metric.
    mse = sm.mean_squared_error(y_test,y_pred)

    # Explained variance score - This score measures how well our model can account for
    # the variation in our dataset. A score of 1.0 indicates that our model is perfect
    evs = sm.explained_variance_score(y_test,y_pred)

    # Random forest regression performance
    print("\n#### Random Forest regressor performance ####")
    print("Mean squared error =", round(mse, 2))
    print("Explained variance score =", round(evs, 2))

    plot_feature_importances(rf_regressor.feature_importances_,'Random Forest regressor', feature_names)