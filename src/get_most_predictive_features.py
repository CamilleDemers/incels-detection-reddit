import pandas as pd
import re
import numpy as np
import multiprocessing
import string

from typing import List

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

# Constants and configuration
CORES = multiprocessing.cpu_count()
base_stopwords = list(stopwords.words('english')) + [
    "'d", "'ll", "'re", "'s", "'ve", '``', 'could', 'might', 'must', "n't", 'need', 'sha', 'wo', 'would'
]
with open ('utils/exclusion_v2.stop', encoding='utf-8') as f:
    additional_stopwords = [x.strip('\n').lower() for x in f.readlines()]

STOPWORDS = base_stopwords + additional_stopwords    

def tokenize_lemmatize_remove_stop_words(text: str) -> List[str]:
    return [wnl.lemmatize(token) for token in word_tokenize(text) if 
            token not in STOPWORDS and
            len(token) > 2 and  # Mots de moins de 2 lettres
            not (bool(re.search(r'\d', token))) and # Mots contenant des chiffres
            not (any(char in string.punctuation for char in token)) # Mots contenant des signes de ponctuation
]

data = pd.read_excel(
    '../data/training_datasets/train_dataset_30pc.xlsx')

dic = {'neutral':0, 'incel': 1}
data['label'] = data['category'].map(dic)

vectorizer = TfidfVectorizer(
    stop_words=STOPWORDS,
    token_pattern=None,
    tokenizer=tokenize_lemmatize_remove_stop_words,
    max_features=15000
)

X = vectorizer.fit_transform(data['text_post'].astype('str'))
y = data['label']

feature_names = vectorizer.get_feature_names_out()

model = LogisticRegression(
    n_jobs=CORES,
)

model.fit(X, y)

coefficients = model.coef_[0]
feature_importance = pd.DataFrame({'Trait': feature_names, 'Coefficient': coefficients})
feature_importance_incels = feature_importance.sort_values('Coefficient', ascending=False)[:20]

feature_importance_neutrals = feature_importance.sort_values('Coefficient', ascending=True)[:20]
feature_importance_neutrals['Coefficient'] = feature_importance_neutrals['Coefficient'].transform(lambda x : np.abs(x))

feature_importance_incels = feature_importance_incels.to_dict('records')
feature_importance_neutrals = feature_importance_neutrals.to_dict('records')

feature_importance = []
for i in range(20):
    feature_importance.append({
        'Trait_incel':feature_importance_incels[i]['Trait'],
        'Coefficient_incel':feature_importance_incels[i]['Coefficient'],
        'Trait_neutre':feature_importance_neutrals[i]['Trait'],
        'Coefficient_neutre':feature_importance_neutrals[i]['Coefficient']
    })


feature_importance = pd.DataFrame(feature_importance)
feature_importance.to_csv('../results/top_features_per_class.csv', index=False)