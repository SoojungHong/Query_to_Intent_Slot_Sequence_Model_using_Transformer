#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import re
from tqdm import tqdm
import json
import pandas as pd
import spacy


tqdm.pandas()
nlp = spacy.load('en_core_web_lg')


def makeDF(path, sample_frac=1.0):
    df = pd.read_csv(path, sep='\t', header=None, names=['q1', 'q2', 'labelled']).sample(frac=sample_frac)
    df['intents'] = df['labelled'].progress_apply(lambda x: re.findall('(?<=IN:)(.*?)(?=\\s)', x))
    df['slots'] = df['labelled'].progress_apply(lambda x: re.findall('(?<=SL:)(.*?)(?=\\s)', x))
    df['top_intent'] = df['intents'].progress_apply(lambda x: x[0])
    df = df[['q1', 'top_intent']]
    return df

def getLabel(l):
    if l in l2idx:
        index = [l2idx[l]]
    else:
        index = [l2idx['<UNKNOWN>']]
    return index


TEST_FILE = "../data/semanticparsingdialog/top-dataset-semantic-parsing/test.tsv"
EVAL_FILE = "../data/semanticparsingdialog/top-dataset-semantic-parsing/eval.tsv"
TRAIN_FILE = "../data/semanticparsingdialog/top-dataset-semantic-parsing/train.tsv"

SAMPLE_FRAC = 1.0

trainDF = makeDF(TRAIN_FILE, sample_frac=SAMPLE_FRAC)
evalDF = makeDF(EVAL_FILE, sample_frac=SAMPLE_FRAC)
testDF = makeDF(TEST_FILE, sample_frac=SAMPLE_FRAC)


l2idx = {'<UNKNOWN>':0}
idx = max(l2idx.values())+1
for i, row in tqdm(pd.concat([trainDF]).iterrows()):
    for t in nlp(row['top_intent']):
        if t.text not in l2idx:
            l2idx[t.text] = idx
            idx += 1


trainXlist = []
trainYlist = []
for i, row in tqdm(trainDF.iterrows()):
    trainXlist.append(nlp(row['q1']).vector)
    trainYlist.append(getLabel(row['top_intent']))
trainX = np.vstack(trainXlist)
trainY = np.vstack(trainYlist)
trainY = trainY.reshape(-1)

evalXlist = []
evalYlist = []
for i, row in tqdm(evalDF.iterrows()):
    evalXlist.append(nlp(row['q1']).vector)
    evalYlist.append(getLabel(row['top_intent']))
evalX = np.vstack(evalXlist)
evalY = np.vstack(evalYlist)
evalY = evalY.reshape(-1)

testXlist = []
testYlist = []
for i, row in tqdm(testDF.iterrows()):
    testXlist.append(nlp(row['q1']).vector)
    testYlist.append(getLabel(row['top_intent']))
testX = np.vstack(testXlist)
testY = np.vstack(testYlist)
testY = testY.reshape(-1)


logreg = LogisticRegression(solver='liblinear')

scaler = StandardScaler().fit(trainX)

logreg.fit(scaler.transform(trainX), trainY)

evalYhat = logreg.predict(scaler.transform(evalX))

report = metrics.classification_report(
    evalY,
    evalYhat,
    labels=[l2idx[k] for k in l2idx],
    target_names=[k for k in l2idx],
    zero_division=1,
    output_dict=True
)

report['accuracy'] = metrics.accuracy_score(evalY, evalYhat)

with open('baseline_report.json', 'w') as f:
    json.dump(report, f)
