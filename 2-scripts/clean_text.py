import pandas as pd
import re
import os
import string

### NLTK : Stopword filtering & tokenization
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

punct = string.punctuation
punct += '’'

stopw = stopwords.words('english')
for w in stopw:
    w_nopunct = re.sub(f'[{punct}]', '', w)
    if w_nopunct not in stopw:
        stopw.append(w_nopunct)

stopw += ["'d", "'ll", "'re", "'s", "'ve", 'could', 'might', 'must', "n't", 'need', 'sha', 'wo', 'would']

def cleanText(text):
    text = re.sub(f'[{punct}]', '', text) 
    text = text.replace('\n', '')
    text = text.lower()
    return text

folder = '../1-data/training_datasets/'
datasets = os.listdir(folder)

for file in datasets:
    ratio = file[14:-7]
    
    df = pd.read_excel(os.path.join(folder, file))

    # Nettoyage
    df['cleaned_text_post'] = df['text_post'].astype(str).apply(cleanText)
    df = df[['cleaned_text_post', 'category']]

    df.to_excel('../2-scripts/1-preprocessing/cleaned_' + file, index=False)