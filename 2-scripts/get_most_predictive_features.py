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
STOPWORDS = list(stopwords.words('english')) + [
    "'d", "'ll", "'re", "'s", "'ve", '``', 'could', 'might', 'must', "n't", 'need', 'sha', 'wo', 'would'
]

def tokenize_lemmatize_remove_stop_words(text: str) -> List[str]:
    return [wnl.lemmatize(token) for token in word_tokenize(text) if 
            token not in STOPWORDS and
            len(token) > 2 and  # Mots de moins de 2 lettres
            not (bool(re.search(r'\d', token))) and # Mots contenant des chiffres
            not (any(char in string.punctuation for char in token)) # Mots contenant des signes de ponctuation
]

data = pd.read_excel(
    '../1-data/training_datasets/train_dataset_30pc.xlsx')

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
feature_importance = feature_importance.sort_values('Coefficient', ascending=False)[:50]

# Create LaTeX table string
latex_table = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{|c|c|}\n\\hline\nTrait & Coefficient \\\\\n\\hline\n"
for row in feature_importance.to_dict('records'):
    latex_table += f"{row['Trait']} & {row['Coefficient']:.4f} \\\\\n"
latex_table += "\\hline\n\\end{tabular}\n\\caption{25 traits les plus prédicitfs de la classe 'incels'}\n\\label{tab:top_features}\n\\end{table}"

#print(latex_table)