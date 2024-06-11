import os
import re
import logging
import multiprocessing
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier

from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
CORES = multiprocessing.cpu_count()
STOPWORDS = list(stopwords.words('english')) + [
    "'d", "'ll", "'re", "'s", "'ve", '``', 'could', 'might', 'must', "n't", 'need', 'sha', 'wo', 'would'
]
DATA_FOLDER = '../1-data/training_datasets'
TEST_DATA_FILE = '../1-data/test_dataset_10.xlsx'
RESULTS_FOLDER = '../3-results'
N_FEATURES_VALUES = [100, 200, 300, 400, 500, 750, 1000, 2500, 5000, 10000, 15000]

SCORING = {
    'accuracy': 'accuracy',
    'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
    'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
    'f1_macro': make_scorer(f1_score, average='macro', zero_division=0)
}

ALGORITHMS = {
    'kNN (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=CORES),
    'Support Vector Machines (SVM)': LinearSVC(dual="auto"),
    'Logistic Regression': LogisticRegression(n_jobs=CORES),
    'Perceptron': Perceptron(),
    'Random Forest': RandomForestClassifier(n_jobs=CORES)
}

# Function definitions
def load_datasets(folder: str) -> List[str]:
    return os.listdir(folder)

def load_test_data(filepath: str) -> pd.DataFrame:
    return pd.read_excel(filepath)

def tokenize_lemmatize_remove_stop_words(text: str) -> List[str]:
    return [wnl.lemmatize(token) for token in word_tokenize(text) if 
            token not in STOPWORDS and
            len(token) > 2 and  # Mots de moins de 2 lettres
            not (bool(re.search(r'\d', token))) and # Mots contenant des chiffres
            not (any(char in string.punctuation for char in token)) # Mots contenant des signes de ponctuation
]

def evaluate_algorithm(algorithm, X, y, cv, scoring) -> Dict[str, Any]:
    return cross_validate(
        estimator=algorithm,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

def save_report(df: pd.DataFrame, filename: str):
    df.to_csv(filename, index=False)

def vectorize_tf_idf(corpus: List[str], corpus_test: List[str], n_features: int):
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_lemmatize_remove_stop_words,
        token_pattern=None,
        max_features=n_features
    )
    X = vectorizer.fit_transform(corpus)
    X_test = vectorizer.transform(corpus_test)
    return X, X_test

def vectorize_word2vec(corpus: List[str], corpus_test: List[str], n_features: int):
    tokenized_corpus = [list(tokenize_lemmatize_remove_stop_words(doc)) for doc in corpus]
    tokenized_corpus_test = [list(tokenize_lemmatize_remove_stop_words(doc)) for doc in corpus_test]
    model_w2v = Word2Vec(
        tokenized_corpus,
        vector_size=n_features,
        workers=CORES
    )
    
    def vectorize(document_tokenized: List[str]) -> np.ndarray:
        words_vecs = [model_w2v.wv[word] for word in document_tokenized if word in model_w2v.wv]
        if len(words_vecs) == 0:
            return np.zeros(n_features)
        words_vecs = np.array(words_vecs)
        return words_vecs.mean(axis=0)
    
    X = np.array([vectorize(doc) for doc in tokenized_corpus])
    X_test = np.array([vectorize(doc) for doc in tokenized_corpus_test])
    
    return X, X_test

def vectorize_doc2vec(corpus: List[str], corpus_test: List[str], n_features: int):
    tokenized_corpus = [list(tokenize_lemmatize_remove_stop_words(doc)) for doc in corpus]
    tokenized_corpus_test = [list(tokenize_lemmatize_remove_stop_words(doc)) for doc in corpus_test]
    tagged_corpus = [TaggedDocument(words, [str(idx)]) for idx, words in enumerate(tokenized_corpus)]
    tagged_corpus_test = [TaggedDocument(words, [str(idx)]) for idx, words in enumerate(tokenized_corpus_test)]
    
    model_dmm = Doc2Vec(
        tagged_corpus,
        dm=1,  # Distributed Memory
        vector_size=n_features,
        workers=CORES
    )
    
    X = [model_dmm.dv[str(doc.tags[0])] for doc in tagged_corpus]
    X_test = [model_dmm.dv[str(doc.tags[0])] for doc in tagged_corpus_test]
    
    return X, X_test

def main(vectorization_method='tfidf'):
    datasets = load_datasets(DATA_FOLDER)
    df_test = load_test_data(TEST_DATA_FILE)
    corpus_test = df_test['text_post'].astype('str')
    y_test = df_test['category'].astype('str')

    training_reports = []
    test_reports = []

    for dataset in datasets:
        ratio = int(dataset[14:-7])
        df = pd.read_excel(os.path.join(DATA_FOLDER, dataset))
        report = []

        corpus = df['text_post'].astype('str')
        y = df['category'].astype('str')

        for n_features in N_FEATURES_VALUES:
            if vectorization_method == 'tfidf':
                X, X_test = vectorize_tf_idf(corpus, corpus_test, n_features)
                vectorization_label = 'TF-IDF'
                ALGORITHMS['Multinomial Naive Bayes'] = MultinomialNB()
            
            elif vectorization_method == 'word2vec':
                if n_features>500:
                    break
                X, X_test = vectorize_word2vec(corpus, corpus_test, n_features)
                vectorization_label = 'CBOW'
                ALGORITHMS['Gaussian Naive Bayes'] = GaussianNB()
            
            elif vectorization_method == 'doc2vec':
                if n_features>500:
                    break
                X, X_test = vectorize_doc2vec(corpus, corpus_test, n_features)
                vectorization_label = 'DM'
                ALGORITHMS['Gaussian Naive Bayes'] = GaussianNB()
            
            else:
                raise ValueError("Unsupported vectorization method")

            for algorithm_name, algorithm in ALGORITHMS.items():
                scores = evaluate_algorithm(algorithm, X, y, StratifiedKFold(shuffle=True, random_state=42), SCORING)

                results = {
                    '% Incels': ratio,
                    'Algorithme': algorithm_name,
                    'Modèle de vectorisation': vectorization_label,
                    'N traits discr.': n_features,
                    'accuracy': scores['test_accuracy'].mean(),
                    'precision': scores['test_precision_macro'].mean(),
                    'recall': scores['test_recall_macro'].mean(),
                    'f1-score': scores['test_f1_macro'].mean()
                }

                report.append(results)
                logger.info(f"{algorithm_name} | {ratio}% Incels | {n_features} traits discr.\n"
                            f"Accuracy: {results['accuracy']:.4f}, Precision: {results['precision']:.4f}, "
                            f"Recall: {results['recall']:.4f}, F1-score: {results['f1-score']:.4f}")

                algorithm.fit(X, y)
                predictions_test = algorithm.predict(X_test)

                test_results = {
                    '% Incels': ratio,
                    'Algorithme': algorithm_name,
                    'Modèle de vectorisation': vectorization_label,
                    'N traits discr.': n_features,
                    'accuracy': accuracy_score(y_test, predictions_test),
                    'precision': precision_score(y_test, predictions_test, average='macro', zero_division=0),
                    'recall': recall_score(y_test, predictions_test, average='macro', zero_division=0),
                    'f1-score': f1_score(y_test, predictions_test, average='macro', zero_division=0)
                }

                test_reports.append(test_results)

        report_df = pd.DataFrame(report)
        report_df['nb_posts_incels'] = (report_df['% Incels'].apply(lambda x: x / 100) * df.shape[0]).astype(int)
        report_df['nb_posts_neutral'] = (report_df['% Incels'].apply(lambda x: 1 - (x / 100)) * df.shape[0]).astype(int)
        report_df['nb_posts_total'] = df.shape[0]

        report_df = report_df[['nb_posts_total', 'nb_posts_incels', 'nb_posts_neutral', '% Incels', 'Algorithme',
                               'Modèle de vectorisation', 'N traits discr.', 'accuracy', 'precision', 'recall', 'f1-score']]

        training_reports.append(report_df)

    final_report_df = pd.concat(training_reports)
    save_report(final_report_df.sort_values(by='f1-score', ascending=False), os.path.join(RESULTS_FOLDER, f'results_training_{vectorization_method}.csv'))

    test_report_df = pd.DataFrame(test_reports)
    save_report(test_report_df.sort_values(by='f1-score', ascending=False), os.path.join(RESULTS_FOLDER, f'results_test_{vectorization_method}.csv'))

if __name__ == '__main__':
    main(vectorization_method='tfidf')  # 'tfidf', 'word2vec' or 'doc2vec'