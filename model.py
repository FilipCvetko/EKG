from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
import ast
import numpy as np
import time

data = pd.read_csv("./pre_model_data/data_5.csv")

labels = data.iloc[:, -1:]
new_data = np.array(data.iloc[:, :-2])
new_labels = np.zeros((21430, 1))

for i, row in enumerate(labels["label"]):
    row = ast.literal_eval(row)
    for j, item in enumerate(row):
        new_labels[i][j] = item

print(labels.iloc[50])
print(new_labels[50])
time.sleep(10)
x_train, x_test, y_train, y_test = train_test_split(new_data, new_labels, test_size=0.1)

x_train = tf.convert_to_tensor(x_train)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)
x_test = tf.convert_to_tensor(x_test)
print(y_test.shape)

model = keras.Sequential([
    Dense(400, activation="relu", input_shape=(5988,)),
    Dense(200, activation="relu"),
    Dense(30, activation="relu"),
    Dense(1, activation="sigmoid")
])

sgd = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.summary()
model.fit(x_train,y_train,validation_split=0.1, epochs=100, batch_size=128)

model.save("./models/model.h5")

results = model.evaluate(x_test, y_test, batch_size=128)

print(results)
