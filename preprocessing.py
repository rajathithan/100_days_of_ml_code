import numpy as np
from sklearn import preprocessing

# Define a array

data = np.array([[3, -1.5, 2, -5.4], [0, 4, -0.3, 2.1], [1, 3.3, -1.9, -4.3]])

# Preprocess the data

# Mean removal - Removing the mean from each feature so that it is centered on 
# zero, this helps in removing the bias from the feature

data_standardized = preprocessing.scale(data)
print("\nMean =", data_standardized.mean(axis=0))
print("Std deviation =", data_standardized.std(axis=0))

# Scaling - The values of each feature in a datapoint can vary between random 
# values. So, sometimes it is important to scale them so that this becomes a 
# level playing field

data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print ("\nMin max scaled data =\n", data_scaled)


# Normalization -Data normalization is used when you want to adjust the values 
# in the feature vector so that they can be measured on a common scale. One of the 
# most common forms of normalization that is used in machine learning adjusts the values 
# of a feature vector so that they sum up to 1

data_normalized = preprocessing.normalize(data, norm='l1')
print("\nL1 normalized data =\n", data_normalized)

# Binarization - Binarization is used when you want to convert your numerical feature 
# vector into a Boolean vector.

data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print("\nBinarized data =\n", data_binarized)

