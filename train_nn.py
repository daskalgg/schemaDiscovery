#!/usr/bin/env python3
# import numpy as np

import tensorflow as tf
from tensorflow import keras
from trainingData import getTrainingData
import math


(data, types, unlabeled) = getTrainingData()

model = keras.models.Sequential()
model.add(
    keras.layers.Dense(len(data[0]), input_shape=(len(data[0]),), activation="relu")
)
model.add(keras.layers.Dense(len(data[0]), activation="relu"))
model.add(keras.layers.Dense(len(types[0])))

print(model.summary())

model.compile(
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=["mae"],
)
train_per = int(len(data) * 0.8)
history = model.fit(
    data[0:train_per],
    types[0:train_per],
    validation_data=(data[train_per:], types[train_per:]),
    epochs=200,
    batch_size=2000,
)

p = model.predict(unlabeled)
# print(unlabeled)
print(p)

import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
accuracy = history_dict["mae"]
val_accuracy = history_dict["val_mae"]

epochs = range(1, len(loss_values) + 1)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

model.save("model_weights")
#
# Plot the model accuracy (MAE) vs Epochs
#
ax[0].plot(epochs, accuracy, "bo", label="Training accuracy")
ax[0].plot(epochs, val_accuracy, "b", label="Validation accuracy")
ax[0].set_title("Training & Validation Accuracy", fontsize=16)
ax[0].set_xlabel("Epochs", fontsize=16)
ax[0].set_ylabel("Accuracy", fontsize=16)
ax[0].legend()
#
# Plot the loss vs Epochs
#
ax[1].plot(epochs, loss_values, "bo", label="Training loss")
ax[1].plot(epochs, val_loss_values, "b", label="Validation loss")
ax[1].set_title("Training & Validation Loss", fontsize=16)
ax[1].set_xlabel("Epochs", fontsize=16)
ax[1].set_ylabel("Loss", fontsize=16)
ax[1].legend()
plt.show()

# The results would be in the range of 0.98 or 0.02 cause we might #still be left with some error rate(which could've been fixed if we #used a bigger training set or different model parameters.
# Since we desire binary results, we just round the results using the #round function of Python.
# The predictions are actually in
# print([x for x in model.predict(test_data)])
