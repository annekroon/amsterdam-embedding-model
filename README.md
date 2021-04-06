# Amsterdam Embedding Model

For the intrinsic evaluation:
'data/raw/question-words.txt': derived from: https://github.com/clips/dutchembeddings


# AEM

This repo attempts to build the 'Amsterdam Embedding Model' (AEM): A news domain specific word embedding model trained on Dutch journalistic content.

Amsterdam Embedding Model (AEM) Corpus

This corpus contains unique sentences derived from a total of 7441914 Dutch news articles that appeared in print media and online sources.
The news articles were derived from the INCA database for the time period: 2000-01-01 - 2017-12-31.

Specifically, news articles appeared in the following sources:

ad (print)                 871156.0
ad (www)                   113671.0
anp                       2048369.0
bd (www)                    14781.0
bndestem (www)              15262.0
destentor (www)             14620.0
ed (www)                    15754.0
fd (print)                 452967.0
frieschdagblad (www)          267.0
gelderlander (www)          10553.0
metro (print)              169362.0
metro (www)                 98307.0
nos                           730.0
nos (www)                   62415.0
nrc (print)                662233.0
nrc (www)                   65885.0
nu                         138084.0
parool (www)                34647.0
pzc (www)                   13312.0
spits (www)                 41422.0
telegraaf (print)          811746.0
telegraaf (www)            307755.0
trouw (print)              603098.0
trouw (www)                 34089.0
tubantia (www)              13779.0
volkskrant (print)         697770.0
volkskrant (www)           129502.0
zwartewaterkrant (www)        378.0


1.657.264.089 raw words and 107.965.966 sentences.

---
This repo contains the following elements:

- [Training] of word embeddings models:

    - We use Word2Vec to train sets of models on the AEM sample with different parameter settings

- Evaluation of these models:

	- Intrinsic evaluation (i.e., syntatic and semantic accuracy of the models), using the following task: [evaluating dutch embeddings](https://github.com/clips/dutchembeddings)

	- Extrinsic evaluation (i.e., performance of the models in downstream tasks)

In doing so, we compare the here-trained models with pre-trained word embedding models on Dutch corpora (i.e., the COW model and a FastText model trained on Wikipedia data, available here: https://github.com/clips/dutchembeddings).

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

- `lib/`: Modules used in python scripts: Classification
- `output/`: Default output directory
- `helpers/`: small scripts to get info on training samples
-`model_training/`: here you will find code used to train the models.

`make_tmpfileuniekezinnen.py` can be used to extract sentences from articles INCA database. Duplicate sentences will be removed right away.
`trainmodel_fromfile_iterator_modelnumber.py` takes a .txt file with sentences as input and trains word2vec models with different parameter settings.


## Data:

The current study tests the quality of classifiers w/wo embedding vectorizers on the following data:

Vermeer, S.: [Dutch News Classifier](https://figshare.com/articles/A_supervised_machine_learning_method_to_classify_Dutch-language_news_items/7314896/1) --> This datasets classifies Dutch news in four topics

Buscher, Vliegenthart & De Vrees: [Policy Issues Classifier](https://www.google.com/search?q=Using+Supervised+Machine+Learning+to+Code+Policy+Issues%3A+Can+Classifiers+Generalize&oq=Using+Supervised+Machine+Learning+to+Code+Policy+Issues%3A+Can+Classifiers+Generalize&aqs=chrome..69i57.688j0j7&sourceid=chrome&ie=UTF-8) --> This paper tests a classifer of 18 topics on Dutch news

We thank the authors of these papers for sharing their data. If there are any issues with the way we handle the data or in case suggestions arise, please contact us.

## Vectorizers:

This projects uses the [embedding vectorizer](https://github.com/ccs-amsterdam/embeddingvectorizer) (credits for Wouter van Atteveld).
