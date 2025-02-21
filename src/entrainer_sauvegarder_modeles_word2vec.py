#!/usr/bin/env python
# coding: utf-8

# Ce script permet d'entraîner et de sauvegarder des plongements Word2Vec de différentes dimensions
# Ces plongements peuvent par la suite être réimportés pour entraîner des classifieurs 

import string
import os
import re
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim.models.word2vec import Word2Vec

punct = string.punctuation.replace('-', '')
stopw = stopwords.words('english') + list(punct)

def tokenize_remove_stop_words(text: str):
    return [token for token in word_tokenize(text) if 
            token.lower() not in stopw and
            len(token) > 2 and  # Mots de moins de 2 lettres
            not (bool(re.search(r'\d', token))) and # Mots contenant des chiffres
            not (any(char in punct for char in token))] # Mots contenant des signes de ponctuation

# Lecture du jeu de données d'entraînement
datasets = os.listdir('data/training_datasets/')

for dataset in datasets:
    train = pd.read_csv('data/training_datasets/' + dataset)
    train['category'] = train['category'].apply(lambda x: 1 if x == 'incel' else 0)
    X_train, y_train = train.text_post.astype('str'), train.category

    incels_ratio = dataset[-8:-6]

    tokens = [list(tokenize_remove_stop_words(doc)) for doc in X_train]
    features_w2v = [100, 200, 300, 400, 500]

    for n_features in features_w2v:
        # On crée un modèle distinct pour chaque nombre de dimensions à tester (features_w2v)
        model = Word2Vec(
                tokens,
                vector_size=n_features
        )

        model.save(f'word2vec_models/w2v_{n_features}_dim_{incels_ratio}pc_incels.model')




