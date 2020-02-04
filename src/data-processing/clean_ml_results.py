#!/usr/bin/env python3
#DEPENDS: run_classifier.py
#CREATES: cleans file
#TITLE: Data cleaner (cleans output of ML algorithms)
#DESCRIPTION:Creates a clean data frame with results of ML on different vectorizers, classifiers, and parameter settings.
#PIPE: TRUE

import pandas as pd

def clean_df_true_pred(results):
    data = pd.DataFrame.from_dict(results)

    predicted = data.predicted.apply(pd.Series).merge(data, right_index = True, left_index = True) \
    .drop(["predicted"], axis = 1).melt(id_vars = ['classifier'], value_name = "Predicted label")

    actual = data.actual.apply(pd.Series).merge(data, right_index = True, left_index = True) \
    .drop(["predicted"], axis = 1).melt(id_vars = ['classifier'], value_name = "Actual label")

    df = pd.merge(predicted, actual, how = 'inner', left_index = True, right_index = True)
    df['Classifier'] = df['classifier_x']
    df = df[df.variable_x != 'actual']
    df = df[['Predicted label', 'Actual label', 'Classifier']]
    
    return df