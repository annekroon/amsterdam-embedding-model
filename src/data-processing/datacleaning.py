#!/usr/bin/env python3
#DEPENDS: /home/anne/tmpanne/AEM_output/baseline_classreport.json, /home/anne/tmpanne/AEM_output/baseline_classreport.json,
#CREATES: None
#TITLE: Data cleaner (cleans output of ML algorithms)
#DESCRIPTION:Creates a clean data frame with results of ML on different vectorizers, classifiers, and parameter settings.
#PIPE: TRUE

import sys
print(sys.stdin.read().upper())

import seaborn as sns
import matplotlib.pyplot as plt
import json
import logging
import pandas as pd

def get_data(file):
    path_to_data='/home/anne/tmpanne/AEM_output/'

    fname_sml = '{}{}'.format(path_to_data, file)
    with open(fname_sml) as handle:
        class_report =  json.loads(handle.read())

    appended_data = []

    for i in class_report:
        data = pd.DataFrame.from_dict(i)
        appended_data.append(data)
    df = pd.concat(appended_data)
    df['indicators'] = df.index
    return df

def merge_baseline_embedding(dataset):
    df_baseline = get_data('baseline_classreport_dataset_{}_embed_size_large.json'.format(dataset))
    df_embeddings = get_data('embeddings_classreport_dataset_{}_embed_size_large.json'.format(dataset))
    df = pd.concat([df_baseline , df_embeddings])
    return df

def clean_data_metrics(dataset):
    df = merge_baseline_embedding(dataset)
    i = ['precision', 'recall', 'support', 'f1-score']
    df[df['indicators'].isin(i)]
    df.reset_index(inplace=True)
    d = df[df['indicators'].isin(i)].groupby(['classifier', 'model', 'indicators', 'vectorizer']).max()
    return d

def clean_data_parameters(dataset):
    i = ['clf__C', 'clf__fit_intercept', 'clf__loss', 'clf__max_iter','clf__alpha', 'clf__penalty', 'clf__gamma', 'clf__kernel', 'clf__max_features']

    df = merge_baseline_embedding(dataset)
    df = df[df['indicators'].isin(i)].groupby(['classifier', 'model', 'indicators', 'vectorizer']).max()
    df.reset_index(inplace=True)
    e = df.groupby(['classifier', 'model','vectorizer']).apply(lambda g: pd.Series(g.parameters.values, index= + g.indicators.astype(str)))

    parameters = pd.DataFrame(e)
    parameters.reset_index(inplace=True)

    parameters['unique_'] = parameters['classifier'] + "_" + parameters['model']  + "_" +  parameters['vectorizer']

    p = parameters.pivot(index='unique_', columns='indicators', values=0)

    parameters = parameters.groupby(['classifier', 'model', 'vectorizer']).max()
    p.reset_index(inplace=True)
    len(p) == len(parameters)
    parameters.reset_index(inplace=True)
    parameters = pd.merge(p, parameters, on='unique_')
    return parameters

def get_cleaned_data(dataset):
    df = pd.merge(clean_data_metrics(dataset), clean_data_parameters(dataset), on=['classifier','model','vectorizer'], how='left')
    df.rename(columns={'index': 'metrics'}, inplace=True)
    print("..........loaded the data frame........")
    return df
