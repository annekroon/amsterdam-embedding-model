# Amsterdam Embedding Model

This repo attempts to build the 'Amsterdam Embedding Model' (AEM): A news domain specific word embedding model trained on Dutch journalistic content.

### Amsterdam Embedding Model (AEM) Corpus

This corpus contains unique sentences derived from a total of 7441914 Dutch news articles that appeared in print media and online sources.
The news articles were derived from the INCA database for the time period: 2000-01-01 - 2017-12-31.

Specifically, news articles appeared in the following sources:


| Outlet              | N articles  |
|------------------------|-----------|
| ad (www)               | 113671  |
| ad (print              | 871156  |
| anp                    | 2048369 |
| bd (www)               | 14781   |
| bndestem (www)         | 15262   |
| destentor (www)        | 14620   |
| ed (www)               | 15754   |
| fd (print)             | 452967  |
| frieschdagblad (www)   | 267     |
| gelderlander (www)     | 10553   |
| metro (print)          | 169362  |
| metro (www)            | 98307   |
| nos                    | 730     |
| nos (www)              | 62415   |
| nrc (print)            | 662233  |
| nrc (www)              | 65885   |
| nu                     | 138084  |
| parool (www)           | 34647   |
| pzc (www)              | 13312   |
| spits (www)            | 41422   |
| telegraaf (print)      | 811746  |
| telegraaf (www)        | 307755  |
| trouw (print)          | 603098  |
| trouw (www)            | 34089   |
| tubantia (www)         | 13779   |
| volkskrant (print)     | 697770  |
| volkskrant (www)       | 129502  |
| zwartewaterkrant (www) | 378     |


*1.657.264.089 raw words and 107.965.966 sentences.*

---
This repo contains the following elements:

- [Training](https://github.com/annekroon/amsterdam-embedding-model/tree/master/model-training) of word embeddings models:

    - We use Word2Vec to train sets of models on the AEM sample with different parameter settings

- Evaluation of these models:

	- Intrinsic evaluation (i.e., Syntatic and semantic accuracy of the models), using the following task: [evaluating dutch embeddings](https://github.com/clips/dutchembeddings)

	- Extrinsic evaluation (i.e., performance of the models in downstream tasks)

In doing so, we compare the here-trained models with pre-trained word embedding models on Dutch corpora (i.e., the COW model and a FastText model trained on Wikipedia data, available here: https://github.com/clips/dutchembeddings).

## First findings:

### Intrinsic Evaluation

[Results](https://github.com/annekroon/amsterdam-embedding-model/blob/master/get-figures-intrinsic.ipynb) indicate that regarding the **intrinsic evaluation**, we find that both in terms of semantic and syntatic accuracy, the AEM outperforms the other embedding models. The best model is trained the following parameter settings: *window size = 10, dimensions = 300, negative sampling = 5*

![Intrinsic evaluation](https://github.com/annekroon/amsterdam-embedding-model/blob/master/output/intrinsic_output_2.png)

### Downstream Evaluation

[Results](https://github.com/annekroon/amsterdam-embedding-model/blob/master/get-results-downstream.ipynb) indicate that regarding the **downstream task** marco f1-scores may be boosted when using a vectorizer based on the AEM when compared to a baseline model (using a count / tfidf vectorizer) as well as other Dutch-language embedding models.

Performance across vectorizers based on AEM, cow, wiki and baseline using Burscher et al dataset:

![Downstream evaluation Burscher](https://github.com/annekroon/amsterdam-embedding-model/blob/master/output/downstream_Burscher.png)

Performance across vectorizers based on AEM, cow, wiki and baseline using Vermeer et al dataset:

![Downstream evaluation Vermeer](https://github.com/annekroon/amsterdam-embedding-model/blob/master/output/downstream_Vermeer.png)
---

## Python scripts:

#### run_classifier.py

This script will execute text classification task, and returns accuracy scores for classification with SGD and ExtraTrees with or without embeddings.

Example usage:

```
python3 run_classifier.py  --word_embedding_path ../folder_with_embedding_models/
--word_embedding_sample_size large  --type_vectorizer Tfidf
--data_path data/dataset_vermeer.pkl --output output/output

```

#### run_intrinsic_evaluation.py
This script returns accuracy scores for instrinc evaluation tasks.

Example usage:

```
python3 run_intrinsic_evaluation.py  --word_embedding_path ../folder_with_embedding_models/
--word_embedding_sample_size large  --output output/output
--path_to_evaluation_data ../model_evaluation/analogies/question-words.txt
```


#### get_results_classifier.py

Transforms pkl output of classification task to tables and figures, and saves the output to `tables_and_figures/`

Example usage:

```
python3 get_results_classifier.py --output tables_figures/ --dataset vliegenthart
```

## Directories:

- `src/`: Modules used in python scripts: Classification
- `output/`: default output directory
- `helpers/`: small scripts to get info on training samples
-`model_training/`: here you will find code used to train the models.

`make_tmpfileuniekezinnen.py` can be used to extract sentences from articles INCA database. Duplicate sentences will be removed right away.
`trainmodel_fromfile_iterator_modelnumber.py` takes a .txt file with sentences as input and trains word2vec models with different parameter settings.


## Data:

The current study tests the quality of classifiers w/wo embedding vectorizers on the following data:

Vermeer, S.: [Dutch News Classifier](https://figshare.com/articles/A_supervised_machine_learning_method_to_classify_Dutch-language_news_items/7314896/1) --> This datasets classifies Dutch news in four topics

Burscher, Vliegenthart & de Vreese: [Policy Issues Classifier](https://www.google.com/search?q=Using+Supervised+Machine+Learning+to+Code+Policy+Issues%3A+Can+Classifiers+Generalize&oq=Using+Supervised+Machine+Learning+to+Code+Policy+Issues%3A+Can+Classifiers+Generalize&aqs=chrome..69i57.688j0j7&sourceid=chrome&ie=UTF-8) --> This paper tests a classifer of 18 topics on Dutch news

We thank the authors of these papers for sharing their data. If there are any issues with the way we handle the data or in case suggestions arise, please contact us.

## Vectorizers:

This projects uses the [embedding vectorizer](https://github.com/ccs-amsterdam/embeddingvectorizer) (credits for Wouter van Atteveld).
