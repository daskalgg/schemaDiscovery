#!/usr/bin/env python3
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLWrapper2
import json


def getTrainingData():
    typePredicate = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

    sparql = SPARQLWrapper("http://83.212.77.24:8890/sparql/")

    # dataSet = "BNF"  # Conference DBpedia BNF
    # dataSet = "Conference"  # Conference DBpedia BNF
    dataSet = "DBpedia"  # Conference DBpedia BNF
    # 1. all trilets
    sparql.setQuery(
        """
        select ?s ?p ?o
        from <http://localhost:8890/"""
        + dataSet
        + """>
        where { ?s ?p ?o}
    """
    )
    sparql.setReturnFormat(JSON)
    tripletsAll = sparql.query().convert()

    # 2. machine learning input layer: all predicates (properties) except type, ordered. Based on predicates of each subject we conclude it s type, ordered
    sparql.setQuery(
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
    sparql.setReturnFormat(JSON)
    allPredicatesButType = sparql.query().convert()

    # 3. types (NN output), machine learning output layer: all objects with predicate ‘type’ ordered. All types ordered. They are the outputs plus one for the ones not classified which will be processed in second phase
    sparql.setQuery(
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
    sparql.setReturnFormat(JSON)
    allTypes = sparql.query().convert()

    types = []
    predicates = []

    for t in allTypes["results"]["bindings"]:
        types.append(t["o"]["value"])

    for p in allPredicatesButType["results"]["bindings"]:
        predicates.append(p["predicate"]["value"])

    # dictionary for all subjects
    trainingSet = {}
    for triplet in tripletsAll["results"]["bindings"]:
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
