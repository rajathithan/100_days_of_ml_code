import numpy as np 
from sklearn import linear_model
import matplotlib.pyplot as plt 


# plot classifier function
def plot_classifier(classifier,data,y):
    # define ranges to plot the figure
    # take the first element in the 2D array for x
    x_min, x_max = min(data[:,0]) - 1.0, max(data[:,0]) + 1.0
    # take the second element in the 2D array for y
    y_min, y_max = min(data[:,1]) - 1.0, max(data[:,1]) + 1.0  
    # step size for the mesh grid
    step_size = 0.01
    # mesh grid is used for identifying the boundary.
    # returns co-ordinate matrics for co-ordinate vectors
    x_values, y_values = np.meshgrid(np.arange(x_min,x_max,step_size),
    np.arange(y_min,y_max,step_size))    
    #compute the classifer output
    # np.c_ - is used for horizontal concatenation of the matrix
    # np.ravel - is used for returning the array of the same type
    mesh_output = classifier.predict(np.c_[x_values.ravel(),y_values.ravel()])
    # reshape the array - gives a new shape to array without 
    # changing data
    mesh_output = mesh_output.reshape(x_values.shape)
    # plot the output
    plt.figure()
    # color scheme
    # https://matplotlib.org/examples/color/colormaps_reference.html
    plt.pcolormesh(x_values,y_values,mesh_output,cmap=plt.cm.spring)
    # scatter plot
    plt.scatter(data[:, 0], data[:, 1], c=y, s=80, edgecolors='black',
    linewidth=1, cmap=plt.cm.Paired)
    # specify the boundaries of the figure
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    # specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(min(data[:, 0])-1), int(max(data[:, 0])+1), 1.0)))
    plt.yticks((np.arange(int(min(data[:, 1])-1), int(max(data[:,1])+1), 1.0)))
    plt.show()
 



if __name__ == '__main__':
    # 2D array
    data = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2] , [1.2, 1.9],
    [6, 2], [5.7, 1.5], [5.4, 2.2]])
    # 1D array , Assigning labels to those datapoints    
    y = np.array([0,0,0,1,1,1,2,2,2])
    # Initialize the logistic regression classifier
    # solver parameter specifies the solver that the algorithm will use to solve
    # the system of equations
    # C parameter , controls the regularization strenght. A lower values indicates
    # higher regularization strength.
    classifier = linear_model.LogisticRegression(solver='liblinear',C=100)
    classifier.fit(data,y)
    plot_classifier(classifier,data,y)

    

