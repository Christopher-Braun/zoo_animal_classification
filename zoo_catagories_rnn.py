import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('zoo.csv')
X = dataset.iloc[:, 1:17].values
y = dataset.iloc[:, -1].values

y[:][y[:]==7]=int(0)
X[:,12][X[:,12]==2]=int(1)
X[:,12][X[:,12]==4]=int(2)
X[:,12][X[:,12]==6]=int(3)
X[:,12][X[:,12]==8]=int(4)

y_som = y

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X12 = onehotencoder.fit_transform(X[:, 12].reshape(-1, 1)).toarray()
y = onehotencoder.fit_transform(y.reshape(-1,1)).toarray()

y = np.asarray(y, dtype = int)
X12 = np.asarray(X12, dtype = int)
Xnew = np.append(X, X12, axis=1)
X = np.delete(Xnew, 12, axis=1)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X_SOM = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 21, sigma = 1.0, learning_rate = 0.5)
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
         markers[y_som[i]],
         markeredgecolor = colors[y_som[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

mappings = som.win_map(X_SOM)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Part 2 - Make the ANN

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
classifier.fit(X_train, y_train, batch_size = 10, epochs = 200)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)

y_pred_cat = np.argmax(y_pred, axis=1)
y_pred_test_cat = np.argmax(y_pred_test, axis=1)

y_train_cat = np.argmax(y_train, axis=1)
y_test_cat = np.argmax(y_test, axis=1)

#Making the Confusion Matrix (compares actual values with predictions)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_cat, y_pred_cat)
cm_test = confusion_matrix(y_test_cat, y_pred_test_cat)

cm_count=0
cm_wrong=0
for i in range(len(cm)):
    cm_count += cm[i,i]
    for v in range(len(cm)):
        cm_wrong += cm[i,v]
cm_wrong -= cm_count
    
cm_test_count=0
cm_test_wrong=0
for i in range(len(cm_test)):
    cm_test_count += cm_test[i,i]
    for v in range(len(cm_test)):
        cm_test_wrong += cm_test[i,v]
cm_test_wrong -= cm_test_count

accuracy = cm_count/(cm_count + cm_wrong)
accuracy_test = cm_test_count/(cm_test_count + cm_test_wrong)


