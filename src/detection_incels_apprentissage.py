import os
import re
import sys
import warnings
import pandas as pd
import argparse
import numpy as np
import torch

from joblib import Parallel, delayed

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from sentence_transformers import SentenceTransformer

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Load a pre-trained SBERT model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2')

# Élimination des avertissements
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

punct = string.punctuation.replace('-', '')
stopw = stopwords.words('english') + list(punct)
stopw += [x.translate
    (str.maketrans('', '', punct)) for x in stopwords.words('english')]

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

def train_and_evaluate(dataset, vectorizer):
    print('Entraînement pour le jeu de données : ', dataset)

    ratio_incels = dataset[-8:-6]

    ### Lecture du jeu de données et partitionnement de en ensembles d'entraînement et de test
    train = pd.read_csv(f'data/training_datasets/{dataset}')
    train['category'] = train['category'].apply(lambda x: 1 if x == 'incel' else 0)

    X_train, y_train = train.text_post.astype('str'), train.category

    ### Définition des fonctions de vectorisation    
    # TF-IDF 
    if vectorizer=='tfidf':
        vectorizers = {
            'TfidfVectorizer' : TfidfVectorizer(            
                stop_words=stopw,
                tokenizer=word_tokenize,
                min_df=2,
                token_pattern=None
            )
        }

        param_grid = [
            {
                "vectorizer__max_features": features_tfidf,
                "classify" : classifiers + [MultinomialNB()]
            }
        ]

    # Word2Vec
    elif vectorizer=='word2vec':
        # Charger les modèles Word2Vec
        word2vec_transformers = [FunctionTransformer(
            lambda x: vectorize_word2vec(
                x, 
                w2v_model = Word2Vec.load(
                    f"word2vec_models/w2v_{i}_dim_{ratio_incels}pc_incels.model")
            )
        ) for i in features_w2v]

        vectorizers = {
            'Word2Vec__100' : word2vec_transformers[0],
            'Word2Vec__200' : word2vec_transformers[1],
            'Word2Vec__300' : word2vec_transformers[2],
            'Word2Vec__400' : word2vec_transformers[3],
            'Word2Vec__500' : word2vec_transformers[4]
        }

        param_grid = [
            {
                "classify" : classifiers
            }
        ]

    # SBERT
    elif vectorizer == 'sbert':
        vectorizers = {
            'SentenceTransformer (all-MiniLM-L6-v2)': FunctionTransformer(
                lambda x: model.encode(
                    x.squeeze().astype(str).values,
                    batch_size=64,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                    device=device)
            )
        }

        param_grid = [
            {
                "classify" : classifiers
            }
        ]


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Définition du pipeline de recherche d'hyperparamètres 
    for vectorizer_name, vectorizer in vectorizers.items():
        pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("classify", "passthrough")
        ])

        grid_search = GridSearchCV(
            pipeline, 
            param_grid=param_grid, 
            verbose=2, 
            cv=cv,
            n_jobs=1,
            refit='f1_macro', 
            scoring=['accuracy', 'recall', 'precision', 'f1_macro']
        )

        print(f'Running GridSearchCV for {vectorizer_name}...')
        grid_search.fit(X_train, y_train)

        # Stocker les résultats
        results_dic = grid_search.cv_results_
        results_dic['Vectorizer'] = vectorizer_name
        results_dic['Ratio incels'] = int(ratio_incels)
        pd.DataFrame(results_dic).to_csv(f'results/results_training/training_{vectorizer_name}_{ratio_incels}pc_5folds.csv', index=False)
        results_training.append(results_dic)


def main(vectorization_method:str):
    # Pour SBERT on utilise le GPU donc on ne parallélise pas
    if vectorization_method=='sbert':
            # Move sbert to GPU
            model = model.to(device)
            if torch.cuda.is_available():
                print('GPU : ', torch.cuda.get_device_name(0))

            for dataset in os.listdir('data/training_datasets'):
                 train_and_evaluate(dataset, vectorization_method)
    
    # Pour TF-IDF et Word2Vec, on exécute en parallèle sur plusieurs cœurs (CPU) avec joblib
    else:
        Parallel(n_jobs=-1, verbose=2)(
            delayed(train_and_evaluate)(dataset, vectorization_method)
            for dataset in os.listdir('data/training_datasets') 
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vecteurs', type=str, help="Méthode de vectorisation à utiliser ('tfidf', 'word2vec' ou 'sbert')")  

    # Parse arguments
    args = parser.parse_args()

    # Appel de la fonction principale avec la le type de vectoriseur souhaité
    main(vectorization_method=args.vecteurs)