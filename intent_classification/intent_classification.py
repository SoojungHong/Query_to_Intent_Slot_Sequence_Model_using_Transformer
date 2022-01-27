#!/usr/bin/env python
# coding: utf-8
# __author__: jorg.frese@here.com


import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import pandas as pd
import re
import spacy
from tqdm import tqdm
import numpy as np
import time
from time import gmtime, strftime
import json

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from intentClassifierModel import LSTMClassifierMiniBatch, LSTMClassifierMiniBatchNoPT


tqdm.pandas()

LOAD_FROM_DISK = 0
WITH_PT = 0

stopwords = ['the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it']

exec_start_time = time.time()

print("loading spacy model...")
start_time = time.time()
if LOAD_FROM_DISK:
    nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "ner"])
else:
    nlp = spacy.load('en_core_web_lg')
print("...took {} seconds".format(time.time() - start_time))


# utility functions
def preprocess(q, lemmatize=True, mask_stops=True, mask_ne=True):
    """
    input: raw query
    returns:
        new lemmatized, lower-cased query string with
        masked stopwords and named entities
    """
    doc = nlp(q)
    newString = q
    if mask_ne:
        for e in reversed(doc.ents):  # reversed to not modify the offsets of other entities when substituting
            start = e.start_char
            end = start + len(e.text)
            ent_label = ''.join(('-', e.label_, '-'))
            newString = newString[:start] + ent_label + newString[end:]
    tokens = []
    for t in nlp(newString):
        if mask_stops:
            if t.text.lower() in stopwords:
                tokens.append('-stop-')  # stopwords are masked, not excluded
            else:
                if lemmatize:
                    tokens.append(t.lemma_.lower())
                else:
                    tokens.append(t.text.lower())
        else:
            if lemmatize:
                tokens.append(t.lemma_.lower())
            else:
                tokens.append(t.text.lower())
    return ' '.join(tokens)

def makeDF(path, sample_frac=1.0):
    df = pd.read_csv(path, sep='\t', header=None, names=['q1', 'q2', 'labelled']).sample(frac=sample_frac)
    df['intents'] = df['labelled'].progress_apply(lambda x: re.findall('(?<=IN:)(.*?)(?=\\s)', x))
    df['slots'] = df['labelled'].progress_apply(lambda x: re.findall('(?<=SL:)(.*?)(?=\\s)', x))
    df['top_intent'] = df['intents'].progress_apply(lambda x: x[0])
    df['query'] = df['q1'].progress_apply(lambda x: preprocess(x, lemmatize=True, mask_stops=True, mask_ne=True))
    df = df[['query', 'top_intent']]
    return df

def getSeq(q):
    indices = []
    for t in nlp(q):
        if t.text in w2idx:
            indices.append(w2idx[t.text])
        else:
            indices.append(w2idx['<oov>'])
    return torch.LongTensor(indices)

def getLabel(l):
    if l in l2idx:
        index = [l2idx[l]]
    else:
        index = [l2idx['<UNKNOWN>']]
    return torch.LongTensor(index)

def pad_collate(batch):
    xx, yy = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    return xx_pad, yy, x_lens


# data
if LOAD_FROM_DISK:
    print("loading data from disk...")
    start_time = time.time()
    testDF = pd.read_hdf('datastore.h5', key='test')
    evalDF = pd.read_hdf('datastore.h5', key='eval')
    trainDF = pd.read_hdf('datastore.h5', key='train')
    print("...took {} seconds".format(time.time() - start_time))
else:
    TEST_FILE = "/mnt/c/Users/frese/Projects/big_plays/ngls/ngls_query/data/semanticparsingdialog/top-dataset-semantic-parsing/test.tsv"
    EVAL_FILE = "/mnt/c/Users/frese/Projects/big_plays/ngls/ngls_query/data/semanticparsingdialog/top-dataset-semantic-parsing/eval.tsv"
    TRAIN_FILE = "/mnt/c/Users/frese/Projects/big_plays/ngls/ngls_query/data/semanticparsingdialog/top-dataset-semantic-parsing/train.tsv"
    
    SAMPLE_FRAC = 1.0
    TRAIN_SAMPLE_FRAC = 1.0
    
    print("making test dataframe...")
    start_time = time.time()
    testDF = makeDF(TEST_FILE, sample_frac=SAMPLE_FRAC)
    testDF.to_hdf('datastore.h5', 'test')
    testDF = pd.read_hdf('datastore.h5', key='test')
    print("...took {} seconds".format(time.time() - start_time))
    print(testDF.sample(10))
    
    print("making eval dataframe...")
    start_time = time.time()
    evalDF = makeDF(EVAL_FILE, sample_frac=SAMPLE_FRAC)
    evalDF.to_hdf('datastore.h5', 'eval')
    evalDF = pd.read_hdf('datastore.h5', key='eval')
    print("...took {} seconds".format(time.time() - start_time))
    print(evalDF.sample(10))
    
    print("making training dataframe...")
    start_time = time.time()
    trainDF = makeDF(TRAIN_FILE, sample_frac=TRAIN_SAMPLE_FRAC)
    trainDF.to_hdf('datastore.h5', 'train')
    trainDF = pd.read_hdf('datastore.h5', key='train')
    print("...took {} seconds".format(time.time() - start_time))
    print(trainDF.sample(10))

print("trainDF: {}, evalDF: {}, testDF: {}".format(len(trainDF), len(evalDF), len(testDF)))


# index dictionaries
if LOAD_FROM_DISK:
    print("loading index dictionaries from disk...")
    start_time = time.time()
    with open('w2idx.json') as f:
        w2idx = json.load(f)
    with open('l2idx.json') as f:
        l2idx = json.load(f)
    with open('idx2l.json') as f:
        idx2l = json.load(f)
    idx2l = {int(k):v for k,v in idx2l.items()}
    print("...took {} seconds".format(time.time() - start_time))
    print("Vocabulary size: {}".format(len(w2idx)))
    print("Label set size: {}".format(len(l2idx)))
else:
    print("building w2idx...")
    start_time = time.time()
    w2idx = {'<pad>': 0, '<oov>':1}
    idx = max(w2idx.values())+1
    for i, row in tqdm(pd.concat([trainDF]).iterrows()):
        for t in nlp(row['query']):
            if t.text not in w2idx:
                w2idx[t.text] = idx
                idx += 1
    with open('w2idx.json', 'w') as f:
        json.dump(w2idx, f)
    with open('w2idx.json') as f:
        w2idx = json.load(f)
    print("...took {} seconds".format(time.time() - start_time))
    print("Vocabulary size: {}".format(len(w2idx)))
    
    print("building l2idx...")
    start_time = time.time()
    l2idx = {'<UNKNOWN>':0}
    idx = max(l2idx.values())+1
    for i, row in tqdm(pd.concat([trainDF]).iterrows()):
        for t in nlp(row['top_intent']):
            if t.text not in l2idx:
                l2idx[t.text] = idx
                idx += 1
    with open('l2idx.json', 'w') as f:
        json.dump(l2idx, f)
    with open('l2idx.json') as f:
        l2idx = json.load(f)
        
    idx2l = {value: key for key, value in l2idx.items()}
    with open('idx2l.json', 'w') as f:
        json.dump(idx2l, f)
    with open('idx2l.json') as f:
        idx2l = json.load(f)
    idx2l = {int(k):v for k,v in idx2l.items()}
    
    print("...took {} seconds".format(time.time() - start_time))
    print("Label set size: {}".format(len(l2idx)))


# embedding weights matrix
if LOAD_FROM_DISK:
    print("loading embedding weights from disk...")
    start_time = time.time()
    wm = np.load('embedding_weights.npy')
    print("...took {} seconds".format(time.time() - start_time))
else:
    if WITH_PT:
        print("building embedding matrix...")
        start_time = time.time()
        wm = np.zeros((len(w2idx), 300))
        for i, word in tqdm(enumerate(w2idx)):
            wm[i] = nlp(word).vector
        assert len(w2idx) == wm.shape[0]
        np.save('embedding_weights.npy', wm)
        wm = np.load('embedding_weights.npy')
        print("...took {} seconds".format(time.time() - start_time))


# pytorch datasets
print("creating pytorch datasets...")
start_time = time.time()
TRAIN_seqs = []
TRAIN_labels = []
for i, row in trainDF.iterrows():
    TRAIN_seqs.append(getSeq(row['query']))
    TRAIN_labels.append(getLabel(row['top_intent']))
ds_list = []
for s, l in list(zip(TRAIN_seqs, TRAIN_labels)):
    ds_list.append(TensorDataset(s.view(1, -1), l.view(1, -1)))
TRAIN_dataset = ConcatDataset(ds_list)
   
TEST_seqs = []
TEST_labels = []
for i, row in testDF.iterrows():
    TEST_seqs.append(getSeq(row['query']))
    TEST_labels.append(getLabel(row['top_intent']))
ds_list = []
for s, l in list(zip(TEST_seqs, TEST_labels)):
    ds_list.append(TensorDataset(s.view(1, -1), l.view(1, -1)))
TEST_dataset = ConcatDataset(ds_list)
    
EVAL_seqs = []
EVAL_labels = []
for i, row in evalDF.iterrows():
    EVAL_seqs.append(getSeq(row['query']))
    EVAL_labels.append(getLabel(row['top_intent']))
ds_list = []
for s, l in list(zip(EVAL_seqs, EVAL_labels)):
    ds_list.append(TensorDataset(s.view(1, -1), l.view(1, -1)))
EVAL_dataset = ConcatDataset(ds_list)
print("...took {} seconds".format(time.time() - start_time))
    

# setting and/or loading model parameters
if LOAD_FROM_DISK:
    with open('params.json') as f:
        params = json.load(f)
else:
    if WITH_PT:
        embedding_size = wm.shape[1]
    else:
        embedding_size = 300
    params = {
        'EMBEDDING_DIM' : embedding_size,
        'HIDDEN_DIM' : 50,
        'NLAYERS' : 2,
        'VOCAB_SIZE' : len(w2idx),
        'LABEL_SIZE' : len(l2idx),
        'DROPOUT' : 0.6
    }
    with open('params.json', 'w') as f:
        json.dump(params, f)
    with open('params.json') as f:
            params = json.load(f)


# dataloaders
TRAIN_dataloader = DataLoader(dataset=TRAIN_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
numTestSamples = len(TEST_dataset)
TEST_dataloader = DataLoader(dataset=TEST_dataset, batch_size=numTestSamples, shuffle=False, collate_fn=pad_collate)


# training and evaluation
# model = LSTMClassifierMiniBatch(params, PT_WEIGHTS=torch.FloatTensor(wm))
model = LSTMClassifierMiniBatchNoPT(params)
train_losses = []
test_losses = []
test_accuracies = []
train_accuracies = []

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

print("training...")
for epoch in tqdm(range(5)):
    start_time = time.time()
    model.train()
    batch_losses = []
    batch_accuracies = []
    for i, batch in enumerate(TRAIN_dataloader):
        model.zero_grad()
        batch_output = model(batch)
        batch_loss = loss_function(batch_output, Tensor(batch[1]).long())
        batch_loss.backward()
        optimizer.step()
        batch_losses.append(batch_loss)
        # getting the training accuracies
        model.eval()
        batch_labels = Tensor(batch[1]).long()
        batch_output_no_dropout = model(batch)
        batch_yhat = torch.argmax(batch_output_no_dropout, 1)
        batch_numCorrects = torch.sum((batch_yhat == batch_labels))
        batch_accuracy = batch_numCorrects.item() / batch[0].size(0)
        batch_accuracies.append(batch_accuracy)
        model.train()

    epoch_train_loss = sum(batch_losses) / len(batch_losses)
    train_losses.append(epoch_train_loss)
    epoch_train_accuracy = sum(batch_accuracies) / len(batch_accuracies)
    train_accuracies.append(epoch_train_accuracy)
    
    model.eval()
    testBatch = next(iter(TEST_dataloader))
    testOutput = model(testBatch)
    testLabels = Tensor(testBatch[1]).long()
    epoch_test_loss = loss_function(testOutput, testLabels)
    test_losses.append(epoch_test_loss)
    yhat = torch.argmax(testOutput, 1)
    numCorrects = torch.sum((yhat == testLabels))
    epoch_test_accuracy = numCorrects.item() / numTestSamples
    test_accuracies.append(epoch_test_accuracy)
    
    print("Epoch {}: {} seconds".format(epoch+1, (time.time() - start_time)))
    print("Training Loss: {}; Test Loss: {}; Test Accuracy: {}; Train Accuracy: {}".format(epoch_train_loss, epoch_test_loss, epoch_test_accuracy, epoch_train_accuracy))


# saving trained model
torch.save(model.state_dict(), 'LSTM_2L_BD.pth')


# evaluate on evaluation data
print("evaluating model on evaluation dataset...")
start_time = time.time()
numEvalSamples = len(EVAL_dataset)
EVAL_dataloader = DataLoader(dataset=EVAL_dataset, batch_size=numEvalSamples, shuffle=False, collate_fn=pad_collate)
# evalModel = model = LSTMClassifierMiniBatch(params, PT_WEIGHTS=torch.FloatTensor(wm))
# evalModel.load_state_dict(torch.load('LSTM_2L_BD.pth'))
model.eval()
evalBatch = next(iter(EVAL_dataloader))
evalOutput = model(evalBatch)
evalLabels = Tensor(evalBatch[1]).long()
yhat = torch.argmax(evalOutput, 1)
numCorrects = torch.sum((yhat == evalLabels))
accuracy = numCorrects.item() / numEvalSamples
pred = pd.Series([idx2l[idx] for idx in yhat.data.numpy()])
labels = pd.Series([idx2l[idx] for idx in evalLabels.data.numpy()])
evalDF.reset_index(drop=True, inplace=True)
evalDF['predicted'] = pred
evalDF['labels'] = labels
print(evalDF.sample(10))
print("Evaluation accuracy: {}".format(accuracy))
print("...took {} seconds".format(time.time() - start_time))


# plotting results
with plt.style.context('ggplot'):

    plt.rcParams['figure.figsize'] = 12, 6
    
    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('training epoch')
    ax1.set_ylabel('loss', color='b')
    ax1.plot(test_losses, label='test loss', color='g')
    ax1.plot(train_losses, label='train loss', color='b')
    ax1.legend(loc='center left')
    
    ax2 = ax1.twinx()
    
    ax2.set_ylabel('accuracy', color='r')
    ax2.plot(test_accuracies, label='test accuracy', color='r')
    ax2.plot(train_accuracies, label='train accuracy', color='y')
    ax2.legend(loc='center right')
    
    plt.subplots_adjust(top=0.9)
    plt.suptitle('losses and accuracy')
    
pdf = matplotlib.backends.backend_pdf.PdfPages('training_eval.pdf')
for fig in range(1, plt.gcf().number + 1):
    pdf.savefig( fig )
pdf.close()

print("program execution took {} seconds".format(time.time() - exec_start_time))
