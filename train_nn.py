#!/usr/bin/env python3
# import numpy as np

import tensorflow as tf
from tensorflow import keras
from trainingData import getTrainingData
from sklearn.cluster import KMeans
import math


def train(data, types, unlabeled):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(30, input_shape=(len(data[0]),), activation="relu"))
    model.add(keras.layers.Dense(len(types[0]), activation="relu"))
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
    model.save(dataSet + "_weights")
    return predict(data, types, unlabeled, model)


def predict(data, types, unlabeled, model):
    print("data: ", len(data))
    print("types", len(types))
    print("unlabeled: ", len(unlabeled))
    p = model.predict(unlabeled)
    p = p.tolist()
    results = [0] * len(unlabeled)
    unsupervised_instances = []
    for i in range(len(p)):
        testp = sorted(p[i], reverse=True)
        if testp[0] / 2 < testp[1] and testp[0] < 1:
            unsupervised_instances.append(i)
        max = 0
        maxi = 0
        for index, value in enumerate(p[i]):
            if value > max:
                maxi = index
                max = value
        results[i] = maxi
    d = model.predict(data)
    d = d.tolist()
    kmeans = KMeans(n_clusters=len(types[0]), n_init=400).fit(d)
    if len(unsupervised_instances) == 0:
        return results
    unsupervised = []
    for i in unsupervised_instances:
        unsupervised.append(p[i])

    y_pred_kmeans = kmeans.predict(unsupervised)
    for i, y in enumerate(y_pred_kmeans):
        results[unsupervised_instances[i]] = y
    return results


dataSet = "Conference"  # Conference DBpedia BNF HistMunic

(data, types, unlabeled) = getTrainingData(dataSet)

result = []
try:
    model = keras.models.load_model(dataSet + "_weights")
    result = predict(data, types, unlabeled, model)
except OSError as e:
    print(e)
    result = train(data, types, unlabeled)
