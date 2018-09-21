import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.neural_network import MLPRegressor
import sys
from mpl_toolkits import mplot3d

inputs = pd.read_csv('sin.csv')
inputs_1 = inputs.iloc[:,4]
inputs_2 = inputs.iloc[:,5]
inputs = pd.concat([inputs_1, inputs_2], axis=1, sort=False)
outputs = pd.read_csv('sin.csv')
X = inputs.values
Y = outputs.iloc[:,6].values
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , random_state = 5)
mlp = MLPRegressor(hidden_layer_sizes = (10,50) , max_iter=100000, activation='tanh',solver = 'lbfgs', random_state=0, learning_rate_init=0.3 , momentum=0.9)
mlp.fit(X_train , Y_train)
pred1 = mlp.predict(X_test)
pred_train = mlp.predict(X_train)
plt.figure("2D Plot") 
plt.plot(Y_test)
plt.plot(pred1)
plt.figure("3D Plot")
ax = plt.axes(projection='3d')
ax.scatter3D(X_train[:,0], X_train[:,1], pred_train, cmap='viridis', edgecolor='none')
ax.scatter3D(X_train[:,0], X_train[:,1], Y_train, cmap='binary', edgecolor='none')
plt.show()
# print("Predicted value =" , pred1)
score = mlp.score(X_test , Y_test)
print("Score =" , score)
print(mlp.predict([[2*3.14 , 2*3.14]]))