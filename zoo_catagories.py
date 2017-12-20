# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('zoo.csv')
X = dataset.iloc[:, 0:17]
y = dataset.iloc[:, -1].values

y[:][y[:]==7]=int(0)

X[:,12][X[:,12]==2]=int(1)
X[:,12][X[:,12]==4]=int(2)
X[:,12][X[:,12]==6]=int(3)
X[:,12][X[:,12]==8]=int(4)

X12 = X[:,12].reshape(-1, 1)

from keras.utils.np_utils import to_categorical
X12 = to_categorical(X[:, 12].reshape(-1,1))

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 0] = labelencoder_X.fit_transform(X.iloc[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X = X.values

X12 = np.asarray(X12, dtype = int)
Xnew = np.append(X, X12, axis=1)
Xfin = np.delete(Xnew, 12, axis=1)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X_SOM = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 17, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X_SOM)
som.train_random(data = X_SOM, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's', 'x', 'o', 's', 'x', 'v']
colors = ['r', 'g', 'b', 'w', 'y', 'c', 'm']
for i, x in enumerate(X_SOM):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

mappings = som.win_map(X_SOM)
# frauds = mappings[(2,1)]
class4 = np.concatenate((mappings[(2,0)], mappings[(4,0)], mappings[(4,8)]), axis = 0)
class4 = sc.inverse_transform(class4)
class3 = np.concatenate((mappings[(1,8)], mappings[(1,9)], mappings[(2,9)], mappings[(3,9)], mappings[(8,3)], mappings[(8,5)], mappings[(9,2)], mappings[(9,3)], mappings[(9,4)]), axis = 0)
class3 = sc.inverse_transform(class3)
class2 = np.concatenate((mappings[(0,2)], mappings[(0,4)], mappings[(0,5)], mappings[(1,5)], mappings[(1,2)], mappings[(2,2)], mappings[(2,3)], mappings[(2,5)], mappings[(3,3)], mappings[(4,2)], mappings[(4,3)], mappings[(4,4)], mappings[(4,5)], mappings[(4,6)], mappings[(2,6)], mappings[(5,3)], mappings[(6,3)]), axis = 0)
class2 = sc.inverse_transform(class2)
class1 = np.concatenate((mappings[(6,9)], mappings[(7,7)], mappings[(7,8)], mappings[(8,7)]), axis = 0)
class1 = sc.inverse_transform(class1)
class0 = np.concatenate((mappings[(3,0)], mappings[(7,0)], mappings[(8,0)], mappings[(7,1)], mappings[(5,9)]), axis = 0)
class0 = sc.inverse_transform(class0)
class5 = np.concatenate((mappings[(0,0)], mappings[(1,0)]), axis = 0)
class5 = sc.inverse_transform(class5)
class6 = mappings[(5,0)]
class6 = sc.inverse_transform(class6)


class0 = class0.astype(int)
class1 = class1.astype(int)
class2 = class2.astype(int)
class3 = class3.astype(int)
class4 = class4.astype(int)
class5 = class5.astype(int)
class6 = class6.astype(int)


y_pred0 = []
for i,j in enumerate(X):
    if j[0] in class0[:,0]:
        y_pred0.append([j[0], y[i]])

y_pred1 = []
for i,j in enumerate(X):
    if j[0] in class1[:,0]:
        y_pred1.append([j[0], y[i]])

y_pred2 = []
for i,j in enumerate(X):
    if j[0] in class2[:,0]:
        y_pred2.append([j[0], y[i]])
        
y_pred3 = []
for i,j in enumerate(X):
    if j[0] in class3[:,0]:
        y_pred3.append([j[0], y[i]])

y_pred4 = []
for i,j in enumerate(X):
    if j[0] in class4[:,0]:
        y_pred4.append([j[0], y[i]])

y_pred5 = []
for i,j in enumerate(X):
    if j[0] in class5[:,0]:
        y_pred5.append([j[0], y[i]])

y_pred6 = []
for i,j in enumerate(X):
    if j[0] in class6[:,0]:
        y_pred6.append([j[0], y[i]])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)'''

from keras.utils.np_utils import to_categorical
y_train_b = to_categorical(y_train)
y_test_b = to_categorical(y_test)
X_train_b = to_categorical(X_train[:, 1:])
X_test_b = to_categorical(X_test[:, 1:])

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 21))

# Adding the second hidden layer
classifier.add(Dense(units = 21, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train_b, batch_size = 32, epochs = 1500)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

