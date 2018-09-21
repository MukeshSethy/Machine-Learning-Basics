# %matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier

fruits = pd.read_table('fruit_data_with_colors.txt')
print (fruits.head())

lookup_fruit_name = dict(zip(fruits.fruit_label.unique() , fruits.fruit_name.unique()))
print (lookup_fruit_name)

X = fruits[['mass' , 'width' , 'height']]
Y = fruits['fruit_label']

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , random_state =2)
# cmap = plt.cm.get_cmap('gnuplot')
# scatter = scatter_matrix(X_train , c = Y_train , marker = 'o' , s=40 , hist_kwds = {'bins':15} , figsize = (12 , 12) , cmap = cmap)


knn = KNeighborsClassifier(n_neighbors = 5)
print (knn.fit(X_train , Y_train))
print (knn.score(X_test , Y_test))

fruit_prediction = knn.predict([[158 , 7.8 , 7.2]])
print (lookup_fruit_name[fruit_prediction[0]])

# print X_train
from matplotlib import cm
cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(X_train , c = Y_train , marker = 'o' , s = 40 , hist_kwds = {'bins':15} , figsize = (8,8) , cmap = cmap)
plt.show()