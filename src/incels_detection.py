#!/usr/bin/env python
# coding: utf-8

import os
import re
import sys
import warnings
import pandas as pd
import numpy as np
import torch

from joblib import Parallel, delayed

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sentence_transformers import SentenceTransformer

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import shapiro
from scipy.stats import levene

# Load a pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Move the model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
if torch.cuda.is_available():
    print('GPU : ', torch.cuda.get_device_name(0))

# Élimination des avertissements
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

punct = string.punctuation.replace('-', '')
stopw = stopwords.words('english') + list(punct)
stopw += [x.translate
    (str.maketrans('', '', punct)) for x in stopwords.words('english')]

stopw +=  ["'d", "'ll", "'re", "'s", "'ve", '``', 'could', 'might', 'must', "n't", 'need', 'sha', 'wo', 'would']

def tokenize_remove_stop_words(text: str):
    return [token for token in word_tokenize(text) if 
            token.lower() not in stopw and
            len(token) > 2 and  # Mots de moins de 2 lettres
            not (bool(re.search(r'\d', token))) and # Mots contenant des chiffres
            not (any(char in punct for char in token))] # Mots contenant des signes de ponctuation

def vectorize_word2vec(corpus, w2v_model):    
    def vectorize(document_tokenized):
        words_vecs = [w2v_model.wv[word] for word in document_tokenized if word in w2v_model.wv]
        if len(words_vecs) == 0:
            return np.zeros(w2v_model.vector_size)
        words_vecs = np.array(words_vecs)
        return words_vecs.mean(axis=0)
    
    tokenized_corpus = [list(tokenize_remove_stop_words(doc)) for doc in corpus]
    X = np.array([vectorize(doc) for doc in tokenized_corpus])
    return X


features_tfidf = [1000, 2500, 5000, 10000, 15000]
features_w2v = [100, 200, 300, 400, 500]

classifiers = [
    LogisticRegression(), 
    LinearSVC(dual="auto"),
    RandomForestClassifier(n_estimators=32)
]

results_training = []
results_test = []

def train_and_evaluate(dataset):
    print('Entraînement pour le jeu de données : ', dataset)

    ratio_incels = dataset[-8:-6]

    ### Lecture du jeu de données et partitionnement de celles-ci en ensembles d'entraînement et de test
    train = pd.read_csv(f'../data/training_datasets/{dataset}')
    train['category'] = train['category'].apply(lambda x: 1 if x == 'incel' else 0)

    X_train, y_train = train.text_post.astype('str'), train.category

    ### Définition des fonctions de vectorisation    
    # Charger les modèles Word2Vec
    word2vec_transformers = [FunctionTransformer(
        lambda x: vectorize_word2vec(
            x, 
            w2v_model = Word2Vec.load(
                f"../word2vec_models/w2v_{i}_dim_{ratio_incels}pc_incels.model")
        )
    ) for i in features_w2v]

    vectorizers = {
        # # TF-IDF 
        # 'TfidfVectorizer' : TfidfVectorizer(            
        #     stop_words=stopw,
        #     tokenizer=word_tokenize,
        #     min_df=2,
        #     token_pattern=None
        # ),

        # Word2Vec 
        # 'Word2Vec__100' : word2vec_transformers[0],
        # 'Word2Vec__200' : word2vec_transformers[1],
        # 'Word2Vec__300' : word2vec_transformers[2],
        # 'Word2Vec__400' : word2vec_transformers[3],
        # 'Word2Vec__500' : word2vec_transformers[4],

        # Sentence-BERT
        'SentenceTransformer (all-MiniLM-L6-v2)': FunctionTransformer(
            lambda x: model.encode(
                x.squeeze().astype(str).values,
                batch_size=64,
                convert_to_numpy=True,
                show_progress_bar=True,
                device=device)
        )
    }

    tf_idf_param_grid = [
        {
            "vectorizer__max_features": features_tfidf,
            "classify" : classifiers + [MultinomialNB()]
        }
    ]

    w2v_param_grid = [
            {
            "classify" : classifiers
        }
    ]

    sbert_param_grid = [
            {
            "classify" : classifiers
        }
    ]

    param_grid = {
        'TfidfVectorizer' : tf_idf_param_grid,
        'Word2Vec__100' : w2v_param_grid,
        'Word2Vec__200' : w2v_param_grid,
        'Word2Vec__300' : w2v_param_grid,
        'Word2Vec__400' : w2v_param_grid,
        'Word2Vec__500' : w2v_param_grid,
        'SentenceTransformer (all-MiniLM-L6-v2)' : sbert_param_grid
    }

    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42) # Si temps de faire des tests d'hypothèse
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Définition du pipeline de recherche d'hyperparamètres 
    for vectorizer_name, vectorizer in vectorizers.items():
        specific_param_grid = param_grid.get(vectorizer_name, {})

        pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("classify", "passthrough")
        ])

        grid_search = GridSearchCV(
            pipeline, 
            param_grid=specific_param_grid, 
            verbose=2, 
            cv=cv,
            n_jobs=1, # Éviter la concurrence des ressources 
            refit='f1_macro', 
            scoring=['accuracy', 'recall', 'precision', 'f1_macro']
        )

        print(f'Running GridSearchCV for {vectorizer_name}...')
        grid_search.fit(X_train, y_train)

        # Stocker les résultats
        results_dic = grid_search.cv_results_
        results_dic['Vectorizer'] = vectorizer_name
        results_dic['Ratio incels'] = int(ratio_incels)
        pd.DataFrame(results_dic).to_csv(f'../results/results_training_{vectorizer_name}_{ratio_incels}pc_3x_repeated-10folds.csv', index=False)
        results_training.append(results_dic)


# Pour exécuter en parallèle sur plusieurs cœurs (TD-IDF)
# Parallel(n_jobs=-1, verbose=2)(
#     delayed(train_and_evaluate)(dataset)
#     for dataset in os.listdir('../data/training_datasets') 
# )

# Pour SBERT, il faut utiliser le GPU
for dataset in os.listdir('../data/training_datasets'):
    train_and_evaluate(dataset)
