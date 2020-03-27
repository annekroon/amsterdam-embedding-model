import sys
from src.analysis import classifier
import logging
import argparse
import json
import pandas as pd
from src.analysis.classifier import *

def get_scores(args):
    a = classifier.classifier_analyzer(path_to_data=args.data_path, path_to_embeddings=args.word_embedding_path, dataset=args.dataset)
    class_report, results = a.gridsearch_with_classifiers()

    fname_accuracy = '{}embeddings_classreport_{}_embed_size_{}.json'.format(args.outputpath, args.dataset.split('.')[0], args.word_embedding_sample_size)
    fname_predictions = '{}embeddings_true_predicted_{}_embed_size_{}.json'.format(args.outputpath, args.dataset.split('.')[0], args.word_embedding_sample_size)

    with open(fname_accuracy, mode = 'w') as fo:
        json.dump(class_report, fo)
        
    df = clean_df_true_pred(results)
    df.to_json(fname_predictions)


def get_scores_baseline(args):
    a = classifier.classifier_analyzer(path_to_data=args.data_path, path_to_embeddings=args.word_embedding_path, dataset=args.dataset)
    class_report, results = a.gridsearch_with_classifiers_baseline()

    fname_accuracy = '{}baseline_classreport_{}_embed_size_{}.json'.format(args.outputpath, args.dataset.split('.')[0], args.word_embedding_sample_size)
    fname_true_predicted = '{}baseline_true_predicted_{}_embed_size_{}.json'.format(args.outputpath, args.dataset.split('.')[0], args.word_embedding_sample_size)

    with open(fname_accuracy, mode = 'w') as fo:
        json.dump(class_report, fo)

    df = clean_df_true_pred(results)
    df.to_json(fname_true_predicted)

if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Compute accuracy score of pretrained word embedding models')
    parser.add_argument('--word_embedding_sample_size', type=str, required=False, default = 'large', help='Size of sample of pretrained word embedding (small or large)')
    parser.add_argument('--word_embedding_path', type=str, required=True, help='Path of pretrained word embedding.')
    parser.add_argument('--data_path', type=str, required=False, default='data/', help='Path of dataset with annotated data to be classified')
    parser.add_argument('--dataset', type=str, required=False, default='dataset_vermeer.pkl', help='Path of dataset with annotated data to be classified')
    parser.add_argument('--outputpath', type=str, required=False, default='output/output', help='Path of output file (CSV formatted classification scores)')
    args = parser.parse_args()

    print('Arguments:')
    print('word_embedding_sample_size:', args.word_embedding_sample_size)
    print('word_embedding_path:', args.word_embedding_path)
    print('data_path:', args.data_path)
    print('dataset:', args.dataset)
    print('outputpath:', args.outputpath)
    print()
    
    get_scores(args)
    get_scores_baseline(args)
