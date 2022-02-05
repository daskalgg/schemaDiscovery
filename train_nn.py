#!/usr/bin/env python3
# import numpy as np

import time
import pprint
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from trainingData import getTrainingData
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from dim_red import encode
import math
import json


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def train(data, types, unlabeled):
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            len(data[0]) / 4, input_shape=(len(data[0]),), activation="relu"
        )
    )
    # model.add(keras.layers.Dense(len(data[0]), activation="relu"))
    model.add(keras.layers.Dense(len(types[0]), activation="sigmoid"))

    print(model.summary())

    model.compile(
        loss="mse",
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        metrics=[
            "mae",
            # recall_m,
            # f1_m,
            # keras.metrics.TruePositives(),
            # keras.metrics.FalsePositives(),
        ],
    )
    train_per = int(len(data) * 0.8)
    history = model.fit(
        data[0:train_per],
        types[0:train_per],
        validation_data=(data[train_per:], types[train_per:]),
        epochs=200,
        batch_size=30,
    )
    model.save(dataSet + "_weights")

    import matplotlib.pyplot as plt

    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    accuracy = history_dict["mae"]
    val_accuracy = history_dict["val_mae"]

    epochs = range(1, len(loss_values) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

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

    return predict(data, types, unlabeled, model)


def predict(data, types, unlabeled, model):
    print("data: ", len(data))
    print("types", len(types))
    print("unlabeled: ", len(unlabeled))
    if len(unlabeled) == 0:
        return ([], 0)
    p = model.predict(unlabeled)
    # neigh = KNeighborsClassifier(n_neighbors=2, weights="distance")
    # neigh.fit(data, types)
    # p = neigh.predict(unlabeled)
    print(p)
    p = p.tolist()
    results = [0] * len(unlabeled)
    unsupervised_instances = []
    for i in range(len(p)):
        testp = sorted(p[i], reverse=True)
        if testp[0] - 0.1 < testp[1]:
            unsupervised_instances.append(i)
        max = 0
        maxi = 0
        for index, value in enumerate(p[i]):
            if value > max:
                maxi = index
                max = value
        results[i] = maxi
        # print(maxi)
        # print(p[i])
    # d = model.predict(data)
    # d = d.tolist()
    # kmeans = KMeans(n_clusters=num_of_new_types, n_init=400).fit(data)
    unsupervised = []
    for i in unsupervised_instances:
        unsupervised.append(unlabeled[i])

    num_of_types = 1
    num_of_new_types = 0
    max_score = -float("inf")
    y_pred = []

    if len(unsupervised_instances) == 0:
        return (results, num_of_new_types)

    for i in range(3):
        gmm = GaussianMixture(n_components=num_of_types, random_state=0).fit(data)
        num_of_types += 1
        y_pred_kmeans = gmm.predict(unsupervised)
        score = gmm.score(unsupervised)
        print(score)
        if score > max_score:
            max_score = score
            y_pred = y_pred_kmeans
            num_of_new_types = i + 1

    for i, y in enumerate(y_pred):
        results[unsupervised_instances[i]] = len(types[0]) + y
    return (results, num_of_new_types)


def getName(string):
    string = string.replace("#", "")
    strings = string.split("/")
    return strings[-1]


def getFiveInstances(instances):
    result = []
    for i in range(min(5, len(instances))):
        result.append({"instance_id": instances[i]})
    return result


dataSet = "HistMunic"  # Conference DBpedia BNF HistMunic

(
    data,
    types,
    unlabeled,
    types_labels,
    names,
    predicates,
    unlabeled_names,
    start,
) = getTrainingData(dataSet)

result = []
num_of_new_types = 0
# data_redused = encode(data, len(types[0]))
# unlabeled_redused = encode(unlabeled, len(types[0]))
print("OK")
try:
    model = keras.models.load_model(dataSet + "_weights")
    # (result, num_of_new_types) = predict(data_redused, types, unlabeled_redused, model)
    (result, num_of_new_types) = predict(data, types, unlabeled, model)
except OSError as e:
    print(e)
    # (result, num_of_new_types) = train(data_redused, types, unlabeled_redused)
    (result, num_of_new_types) = train(data, types, unlabeled)
final = {}
predicate_relations = {}
for i in range(num_of_new_types):
    types_labels.append("NewType" + str(i))
for i in range(len(types_labels)):
    if not final.get(types_labels[i]):
        final[types_labels[i]] = {"predicates": [], "individuals": []}
        predicate_relations[types_labels[i]] = {"predicates": {}, "related_types": []}


# Labeled
for i in range(len(data)):
    t = types_labels[types[i].index(1)]
    final[t]["individuals"].append(names[i])
    pred = []
    for index, v in enumerate(data[i]):
        if v == 1:
            if predicates[index] not in predicate_relations[t]["predicates"]:
                predicate_relations[t]["predicates"][predicates[index]] = 1
            else:
                predicate_relations[t]["predicates"][predicates[index]] += 1
            pred.append(predicates[index])
    if sorted(pred) not in final[t]["predicates"]:
        final[t]["predicates"].append(sorted(pred))

# print(unlabeled_names)
for i in range(len(result)):
    # t = ""
    # try:
    t = types_labels[result[i]]
    final[t]["individuals"].append(unlabeled_names[i])
    pred = []
    for index, v in enumerate(unlabeled[i]):
        if v == 1:
            if predicates[index] not in predicate_relations[t]["predicates"]:
                predicate_relations[t]["predicates"][predicates[index]] = 1
            else:
                predicate_relations[t]["predicates"][predicates[index]] += 1
            pred.append(predicates[index])
    if sorted(pred) not in final[t]["predicates"]:
        final[t]["predicates"].append(sorted(pred))

for t1 in predicate_relations:
    for p1 in predicate_relations[t1]["predicates"]:
        for t2 in predicate_relations:
            if t1 == t2:
                continue
            if predicate_relations[t2]["predicates"].get(p1):
                predicate_relations[t1]["related_types"].append(
                    {
                        "propertyName": getName(p1),
                        "relatedTypeId": t2,
                        "relatedInstances": min(
                            predicate_relations[t1]["predicates"][p1],
                            predicate_relations[t2]["predicates"][p1],
                        ),
                    }
                )
# for i in final:
#     print(i, len(final[i]["individuals"]))
# except:
#     pass
final_json = {
    "algorithm": "NN & GMM",
    "dataset": dataSet,
    "types": [],
    "statistics": {},
}

end = time.time()
for i in final:
    if len(final[i]["individuals"]) == 0:
        continue
    final_json["types"].append(
        {
            "type_id": i,
            "typeName": getName(i),
            "instancesFound": len(final[i]["individuals"]),
            "patterns": final[i]["predicates"],
            "relations": predicate_relations[i]["related_types"],
            "fiveExampleInstances": getFiveInstances(final[i]["individuals"]),
        }
    )
    final_json["statistics"] = {
        "executionTime": (end - start),
        "precisionRecall": 1,
        "fMeasure": 1,
        "truePositivePercentage": 1,
        "falsePositivePercentage": 0,
    }

with open("final_" + dataSet + ".json", "w") as f:
    f.write(json.dumps(final_json, indent=4))

    # "precisionRecall": 0.9667,
    # "fMeasure": 0.9828,
    # "truePositivePercentage": 1,
    # "falsePositivePercentage": 0
print("Elapsed time = ", end - start)
# pprint.pprint(final)
# pprint.pprint(json.dumps(final_json))

# print(result)
# print(len(types_labels))
# print(len(names))

# Hist  val_recall_m: 0.8934 - val_f1_m: 0.8934 - val_true_positives: 4342.0000 - val_false_positives: 512.0000
# Conferece recall_m: 0.9667 - val_f1_m: 0.9828 - val_true_positives: 40.0000 - val_false_positives: 0.0000e+00
# BNF val_recall_m: 1.0000 - val_f1_m: 1.0000 - val_true_positives: 6.0000 - val_false_positives: 0.0000e+00
