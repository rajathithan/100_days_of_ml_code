# The confusion matrix helps us to understand the performance of
# the classification model. This helps us to understand on how we 
# classify the testing data set into different classes 
# It will help us in fine tuning the algorithms before we make the 
# changes.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Show confusion matrix
def plot_confusion_matrix(confusion_mat):
    # Display data as an image in regular raster.
    plt.imshow(confusion_mat, interpolation='nearest', cmap='hot')
    # title of the plot.
    plt.title('Confusion matrix')
    plt.colorbar()
    #for four distinct labels in the dataset 0,1,2,3
    tick_marks = np.arange(4)
    #print(tick_marks)
    #plot
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



if __name__== '__main__':
    y_true = [1, 0, 0, 2, 1, 0, 3, 3, 3]
    y_pred = [1, 1, 0, 2, 1, 0, 1, 3, 3]
    confusion_mat = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(confusion_mat)
    # Print classification report
    target_names = ['Class-0', 'Class-1', 'Class-2', 'Class-3']
    print(classification_report(y_true, y_pred, target_names=target_names))