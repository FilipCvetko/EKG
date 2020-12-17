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

data = pd.read_csv("./pre_model_data/data_1.csv")
labels = pd.read_csv("./pre_model_data/labels_1.csv")

x_train, x_test, y_train, y_test = train_test_split(data, labels.T, test_size=0.1)

x_train = tf.convert_to_tensor(x_train)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)
x_test = tf.convert_to_tensor(x_test)
print(y_test.shape)

model = keras.Sequential([
    Dense(100, activation="relu", input_shape=(50,)),
    Dense(30, activation="relu"),
    Dense(30, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

model.summary()
time.sleep(10)
model.fit(x_train,y_train,validation_split=0.1, epochs=500)

model.save("./models/model.h5")