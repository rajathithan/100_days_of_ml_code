import numpy as np
from sklearn import preprocessing


ldata = (["cat","dog","cow","cow","pig","pig","pig","horse","cat"])
print("\nCategorical Data for label encoder: \n",ldata)

# labelEncoder
lencoder = preprocessing.LabelEncoder()
lencoder.fit(ldata)

# Label encoder gives a value to the category
print("\n The Distinct values are :",list(lencoder.classes_))

# The numerical values are given based on the alphabetical order of the 
# strings
print("\n The Integer values are :",lencoder.transform(ldata))



ohdata = (["apple","chicken","brocoli"])
# oneHotEncoder - converting categorical values into binary values

ohencoder = preprocessing.OneHotEncoder()
xdata = list(enumerate(ohdata))
ydata = np.array(xdata)
i = np.argsort([1,0])
cdata = ydata[:,i]
print("\nCategorical Data for one hot encoder: \n",cdata)

ohencoder.fit(cdata)
# The binary values for each category is created 
print("\n The Binary values are :\n",ohencoder.transform(cdata).toarray())


unique, counts = np.unique(ldata, return_counts=True)
ohedata = np.asarray((unique, counts)).T
print("\nCategorical Data for one hot encoder:\n",ohedata)

ohencoder.fit(ohedata)

# # The binary values for each category is created 
print("\n The Binary values are :\n",ohencoder.transform(ohedata).toarray())
