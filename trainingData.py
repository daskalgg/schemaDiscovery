#!/usr/bin/env python3
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLWrapper2
import json


def getAll(query):
    finalRes = []
    prevSize = 10000
    sparql = SPARQLWrapper("http://83.212.77.24:8890/sparql/")
    sparql.setReturnFormat(JSON)
    while prevSize == 10000:
        sparql.setQuery(query + " offset " + str(len(finalRes)))
        res = sparql.query().convert()
        finalRes = finalRes + res["results"]["bindings"]
        prevSize = len(res["results"]["bindings"])
    return finalRes


def getTrainingData(dataSet):
    typePredicate = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

    tripletsAll = getAll(
        """
        select ?s ?p ?o
        from <http://localhost:8890/"""
        + dataSet
        + """>
        where { ?s ?p ?o}
    """
    )

    allPredicatesButType = getAll(
        """
    SELECT distinct  ?predicate
    from <http://localhost:8890/"""
        + dataSet
        + """>
    WHERE
    {
      ?subject  ?predicate ?object.
    FILTER (?predicate!= <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
    }
    """
    )

    allTypes = getAll(
        """
        SELECT distinct ?o
        from <http://localhost:8890/"""
        + dataSet
        + """>
        WHERE
        {
        ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?o.
        }
    """
    )

    types = []
    predicates = []

    for t in allTypes:
        types.append(t["o"]["value"])

    for p in allPredicatesButType:
        predicates.append(p["predicate"]["value"])

    # dictionary for all subjects
    trainingSet = {}
    for triplet in tripletsAll:
        # If this subject isn't in the dict, add it
        if not trainingSet.get(triplet["s"]["value"]):
            trainingSet[triplet["s"]["value"]] = {
                "input": [0] * len(predicates),
                "output": [0] * len(types),
                "classified": False,
            }

        # set the correct type to 1
        if triplet["p"]["value"] == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
            trainingSet[triplet["s"]["value"]]["output"][
                types.index(triplet["o"]["value"])
            ] = 1
            trainingSet[triplet["s"]["value"]]["classified"] = True

        # set the correct predicates to 1
        else:
            trainingSet[triplet["s"]["value"]]["input"][
                predicates.index(triplet["p"]["value"])
            ] = 1

    data = []
    types = []
    unlabeled = []
    for t in trainingSet:
        if trainingSet[t]["classified"] == True:
            data.append(trainingSet[t]["input"])
            types.append(trainingSet[t]["output"])
        else:
            unlabeled.append(trainingSet[t]["input"])

    return data, types, unlabeled
