# import the required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score as cvs
from logistic_reg_classification import plot_classifier

# Text file with input data
input_file = 'data_multivar.txt'

# Define the empty arrays
x = []
y = []

# Load the data to x and y arrays
with open(input_file, 'r') as f:
    try:
        for line in f.readlines():
            data = [float(x) for x in line.split(',')]
            x.append(data[:-1])
            y.append(data[-1]) 
    finally:
        f.close()

# convert them to numpy arrays
x = np.array(x)
print(x)
y = np.array(y)
print(y)


# Bayesian classifer - probalitic interpretation - Probability of an event 
# based on the prior knowledge of conditions that might be related to the 
# event
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(x, y)
y_pred = classifier_gaussiannb.predict(x)

# compute accuracy of the classifier
accuracy = 100.0 * (y == y_pred).sum() / x.shape[0]
print("Accuracy of the classifier =", round(accuracy, 2), "%")

plot_classifier(classifier_gaussiannb, x, y)

###############################################
# Train test split

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.25, random_state=5)
classifier_gaussiannb_new = GaussianNB()
classifier_gaussiannb_new.fit(x_train, y_train)
y_test_pred = classifier_gaussiannb_new.predict(x_test)

# compute accuracy of the classifier
accuracy = 100.0 * (y_test == y_test_pred).sum() / x_test.shape[0]
print("Accuracy of the classifier =", round(accuracy, 2), "%")

plot_classifier(classifier_gaussiannb_new, x_test, y_test)

###############################################
# Cross validation and scoring functions

num_validations = 5

# https://scikit-learn.org/stable/modules/model_evaluation.html
accuracy = cvs(classifier_gaussiannb,x, y, scoring='accuracy', 
cv=num_validations)
print("Accuracy: " + str(round(100*accuracy.mean(), 2)) + "%")

# precision is calculated by total number of correct identifications
# divided by the total number of identifications.
precision = cvs(classifier_gaussiannb,x, y, scoring='precision_weighted',
cv=num_validations)
print("Precision: " + str(round(100*precision.mean(), 2)) + "%")

# recall is calculated by total number of correct identifications
# divided by the total number of interesting items in the dataset
recall = cvs(classifier_gaussiannb,x, y, scoring='recall_weighted',
cv=num_validations)
print("Recall: " + str(round(100*recall.mean(), 2)) + "%")

# A good machine learning model should have good precision 
# and good recall percentage . To quantify this we can use the F1 score
# F1 score = 2 * precision * recall / (precision + recall )

f1 = cvs(classifier_gaussiannb,x, y, scoring='f1_weighted', 
cv=num_validations)
print("F1: " + str(round(100*f1.mean(), 2)) + "%")

