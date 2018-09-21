# %matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.neural_network import MLPClassifier

fruits = pd.read_table('fruit_data_with_colors.txt')
print (fruits.head())

lookup_fruit_name = dict(zip(fruits.fruit_label.unique() , fruits.fruit_name.unique()))
print (lookup_fruit_name)

X = fruits[['mass' , 'width' , 'height']]
Y = fruits['fruit_label']

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , random_state =2)
# cmap = plt.cm.get_cmap('gnuplot')
# scatter = scatter_matrix(X_train , c = Y_train , marker = 'o' , s=40 , hist_kwds = {'bins':15} , figsize = (12 , 12) , cmap = cmap)


mlp = MLPClassifier(hidden_layer_sizes = (100,150) , max_iter=100000, activation='tanh',solver = 'lbfgs', random_state=0, learning_rate_init=0.3 , momentum=0.9)
print (mlp.fit(X_train , Y_train))
print ("Score =" , mlp.score(X_test , Y_test))

fruit_prediction = mlp.predict([[162 , 7.5 , 7.5]])
print (lookup_fruit_name[fruit_prediction[0]])

# print X_train
from matplotlib import cm
cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(X_train , c = Y_train , marker = 'o' , s = 40 , hist_kwds = {'bins':15} , figsize = (8,8) , cmap = cmap)
# plt.show()