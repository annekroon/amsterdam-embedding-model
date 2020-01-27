#!/usr/bin/env python3
#DEPENDS: /home/anne/tmpanne/AEM_output/baseline_classreport.json, /home/anne/tmpanne/AEM_output/baseline_classreport.json,
#CREATES: none
#TITLE: Data cleaner (cleans output of ML algorithms)
#DESCRIPTION: data cleaner
#PIPE: TRUE

import sys
print(sys.stdin.read().upper())


iimport seaborn as sns
import matplotlib.pyplot as plt
import json
import logging
import pandas as pd

def get_data(file):
    path_to_data='/home/anne/tmpanne/AEM_output/'
    file = file

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

def merge_baseline_embedding():
    df_baseline = get_data('baseline_classreport.json')
    df_embeddings = get_data('embeddings_classreport.json')
    df = pd.concat([df_baseline , df_embeddings])
    return df

def clean_data_metrics():
    df = merge_baseline_embedding()
    i = ['precision', 'recall', 'support', 'f1-score']
    df[df['indicators'].isin(i)]
    df.reset_index(inplace=True)
    d = df[df['indicators'].isin(i)].groupby(['classifier', 'model', 'indicators', 'vectorizer']).max()
    return d

def clean_data_parameters():
    i = ['clf__C', 'clf__fit_intercept', 'clf__loss', 'clf__max_iter','clf__alpha', 'clf__penalty', 'clf__gamma', 'clf__kernel', 'clf__max_features']

    df = merge_baseline_embedding()
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

def get_cleaned_data():
    return pd.merge(clean_data_metrics(), clean_data_parameters(), on=['classifier','model','vectorizer'], how='left')
