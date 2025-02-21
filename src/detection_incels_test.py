import os
import sys
import warnings
import pandas as pd
import torch

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import FunctionTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

# Load pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
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


# Initialisation des variables pour stocker les résultats par classe
results_macro = []
results_incels = []
results_not_incels = []

# Évaluation des 20 meilleurs modèles sur des données inédites
for vectorizer in ['tfidf', 'sbert']:
        if vectorizer == 'tfidf':
                ratio_incels = [40, 50, 60]
                n_features = [10000, 15000]

        elif vectorizer == 'sbert':
                ratio_incels = [20, 30, 40, 50, 60, 70, 80]
                n_features = [384]          

        for ratio_incel in ratio_incels:
            ### Lecture des jeux de données d'entraînement et de test 
            train = pd.read_csv(f'data/training_datasets/train_dataset_{ratio_incel}pc.csv')
            test = pd.read_csv('data/test_dataset_10pc.csv')

            train['category'] = train['category'].apply(lambda x: 1 if x == 'incel' else 0)
            test['category'] = test['category'].apply(lambda x: 1 if x == 'incel' else 0)

            X, y_train = train.text_post.astype('str'), train.category
            X_test, y_test = test.text_post.astype('str'), test.category

            for n in n_features:
                if vectorizer == 'tfidf':
                    classifiers = [    
                            LogisticRegression(), 
                            LinearSVC(dual="auto"),
                            MultinomialNB()
                    ] 

                    vectorizer = TfidfVectorizer(            
                        stop_words=stopw,
                        tokenizer=word_tokenize,
                        min_df=2,
                        max_features=n,
                        token_pattern=None
                    )

                if vectorizer == 'sbert':
                    classifiers = [    
                        LogisticRegression(), 
                        LinearSVC(dual="auto"),
                        RandomForestClassifier(n_estimators=32)
                    ]

                    vectorizer =  FunctionTransformer(
                        lambda x: SentenceTransformer('all-MiniLM-L6-v2').encode(
                                x.squeeze().astype(str).values,
                                batch_size=64,
                                convert_to_numpy=True,
                                show_progress_bar=True,
                                device=device)
                    ) 

                X_train = vectorizer.fit_transform(X)
                X_test = vectorizer.transform(X_test)
                
                for classifier in classifiers:
                        model = classifier.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        res = classification_report(
                                y_test, 
                                y_pred, 
                                target_names=['not incel', 'incel'],
                                output_dict=True
                        )

                        res_macro, res_incel, res_not_incel = res['macro avg'], res['incel'], res['not incel']
                        res_macro['accuracy'] = res_incel['accuracy'] = res_not_incel['accuracy'] = res['accuracy']
                        res_macro['vectorizer'] = res_incel['vectorizer'] = res_not_incel['vectorizer'] = vectorizer
                        res_macro['ratio_incels'] = res_incel['ratio_incels'] = res_not_incel['ratio_incels'] = ratio_incel
                        res_macro['n_features'] = res_incel['n_features'] = res_not_incel['n_features'] = n
                        res_macro['classifier'] = res_incel['classifier'] = res_not_incel['classifier'] = type(classifier).__name__

                        results_macro.append(res_macro)
                        results_incels.append(res_incel)
                        results_not_incels.append(res_not_incel)
                        

                X_test = test.text_post.astype('str')
    
pd.DataFrame(results_macro).to_csv('results/results_test/test_sbert_tfidf_macro_avg.csv', index=False)
pd.DataFrame(results_incels).to_csv('results/results_test/test_sbert_tfidf_classe_incel.csv', index=False)
pd.DataFrame(results_not_incels).to_csv('results/results_test/test_sbert_tfidf_classe_not_incel.csv', index=False)