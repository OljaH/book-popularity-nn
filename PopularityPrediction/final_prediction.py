from __future__ import print_function
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df_test = pd.read_excel("te_m.xlsx")
df_train = pd.read_excel("tr_m.xlsx")
textSet=False
if textSet:
    relevant_features=['vggClass0', 'vggClass1', 'vggClass2', 'vggClass3', 'vggClass4', 'vggClass5',
                        'vggClass6', 'vggClass7', 'vggClass8', 'vggClass9', 'vggClass10',
                        'doc2vec0', 'doc2vec1', 'doc2vec2', 'doc2vec3', 'doc2vec4',
                        'doc2vec5', 'doc2vec6', 'doc2vec7', 'doc2vec8', 'doc2vec9',
                        'sentiment']
relevant_features = ['born', 'died', 'influences', 'website', 'awardsNo', 'datePublished', 'pages', 'rating',
                     'yearWritten', 'publisher_Grand Central Publishing', 'publisher_Scholastic',
                     'publisher_Grosset & Dunlap','author_Literature & Fiction', 'author_Fiction',
                     'author_Mystery & Thrillers', "author_Children's Books",'author_Poetry',
                     'author_Historical Fiction', 'books_Fiction', 'books_Mystery', 'books_Fantasy',
                     'books_Classics', 'books_Nonfiction', 'books_Childrens', 'books_Literature',
                     'books_Young Adult', 'books_Science Fiction', 'books_Thriller','books_Crime',
                     'books_Religion', 'topic_summary_11', 'topic_summary_13', 'topic_summary_18',
                     'topic_summary_0', 'sentiment', 'vggClass0', 'vggClass1', 'vggClass2', 'vggClass3', 'vggClass4', 'vggClass5',
                        'vggClass6', 'vggClass7', 'vggClass8', 'vggClass9', 'vggClass10',
                        'doc2vec0', 'doc2vec1', 'doc2vec2', 'doc2vec3', 'doc2vec4',
                        'doc2vec5', 'doc2vec6', 'doc2vec7', 'doc2vec8', 'doc2vec9',]

y_train = df_train.pop('rating')
X_train = df_train.loc[:, relevant_features]

X_test = df_test.loc[:, relevant_features]
y_test = df_test.pop('rating')
# X_train_nn, X_validation, y_train_nn, y_validation = train_test_split(X_train_nn_1, y_train_nn_1, test_size=0.25, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=777)
# scale units
y_train_nn = np.nan_to_num(y_train)
X_train_nn = np.nan_to_num(X_train)
y_val_nn = np.nan_to_num(y_val)
X_val_nn = np.nan_to_num(X_val)

X_test_nn = np.nan_to_num(X_test)
y_test_nn = np.nan_to_num(y_test)


# Start neural network
network = Sequential()

# Add fully connected layer with a ReLU activation function
network.add(Dense(units=64, activation='sigmoid', input_shape=(X_train_nn.shape[1],)))

# Add fully connected layer with a ReLU activation function
network.add(Dense(units=32, activation='sigmoid'))


# Add fully connected layer with no activation function
network.add(Dense(units=10))
network.add(Dense(1, kernel_initializer='normal'))
# Compile neural network
network.compile(loss='mean_squared_error', # Mean squared error
                optimizer='adam', # Optimization algorithm
                metrics=['mse']) # Mean squared error


# Train neural network
history = network.fit(X_train_nn, # Features
                      y_train_nn, # Target vector
                      epochs=200, # Number of epochs
                      verbose=2, # No output
                      batch_size=64, # Number of observations per batch
                   validation_data=(X_val_nn, y_val_nn)) # Data for evaluation

predictions = network.predict(np.nan_to_num(X_test))
# Calculate the absolute errors
errors = abs(predictions - np.nan_to_num(y_test))
import matplotlib.pyplot as plt
plt.scatter(y_test,predictions)
# plt.plot(X_test_nn, y_test_nn)
plt.xlim(2, 5)
plt.ylim(2, 5)
plt.xlabel("real values")
plt.ylabel("predictions")
plt.show()
print('Mean Squared Error:', round(np.mean(errors), 2))


