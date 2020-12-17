from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
import ast
import numpy as np
import time

file = h5py.File("./pca_data/processed.h5", "r")
data = file["processed"]
new_data = np.array(data)

labels = pd.read_csv("./csv/labels_included.csv")
labels = labels["diagnostic_superclass"]

arr = np.zeros((21430, 5))
for a, item in enumerate(labels):
    item = ast.literal_eval(item)
    for b, j in enumerate(item):
        arr[a][b] = j

x_train, x_test, y_train, y_test = train_test_split(new_data, arr, test_size=0.1)

x_train = tf.convert_to_tensor(x_train)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)
x_test = tf.convert_to_tensor(x_test)

model = keras.Sequential([
    Dense(100, activation="selu", input_shape=(50,)),
    Dense(30, activation="selu"),
    Dense(30, activation="selu"),
    Dense(5, activation="sigmoid")
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

model.fit(x_train,y_train,validation_split=0.1, epochs=500)

model.save("./models/model.h5")