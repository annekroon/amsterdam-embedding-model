import sys  
from src.analysis import classifier
import logging
import argparse
import json
import pandas as pd
from src.analysis.classifier import *

path_to_data='/home/anne/tmpanne/AEM_data/'
#dataset = 'dataset_vermeer.pkl'
dataset = 'dataset_burscher.pkl'
outputpath = '/home/anne/tmpanne/AEM_output/'
path_to_embeddings = '/home/anne/tmpanne/fullsample/'

def get_scores(path_to_data, path_to_embeddings, dataset, outputpath):
    a = classifier.classifier_analyzer(path_to_data=path_to_data, path_to_embeddings=path_to_embeddings, dataset=dataset)
    class_report, results = a.gridsearch_with_classifiers()

    fname_accuracy = '{}embeddings_classreport_{}.json'.format(outputpath, dataset)
    fname_predictions = '{}embeddings_true_predicted_{}.json'.format(outputpath, dataset)

    with open(fname_accuracy, mode = 'w') as fo:
        json.dump(class_report, fo)
        
    df = clean_df_true_pred(results)
    df.to_json(fname_predictions)


def get_scores_baseline(path_to_data, path_to_embeddings, dataset, outputpath):
    a = classifier.classifier_analyzer(path_to_data=path_to_data, path_to_embeddings=path_to_embeddings, dataset=dataset)
    class_report, results = a.gridsearch_with_classifiers_baseline()

    fname_accuracy = '{}baseline_classreport_{}.json'.format(outputpath, dataset)
    fname_true_predicted = '{}baseline_true_predicted_{}.json'.format(outputpath, dataset)

    with open(fname_accuracy, mode = 'w') as fo:
        json.dump(class_report, fo)

    df = clean_df_true_pred(results)
    df.to_json(fname_true_predicted)

    
if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    
    get_scores(path_to_embeddings=path_to_embeddings,path_to_data=path_to_data, dataset=dataset, outputpath = outputpath)
    get_scores_baseline(path_to_embeddings=path_to_embeddings,path_to_data=path_to_data, dataset=dataset, outputpath=outputpath)