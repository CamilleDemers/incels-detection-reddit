import pandas as pd
import re
import os
import string

punct = string.punctuation
punct += '’'

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

dataset_test = '../1-data/test_dataset_10pc.xlsx'
ratio = '10'
df = pd.read_excel(dataset_test)

# Nettoyage
df['cleaned_text_post'] = df['text_post'].astype(str).apply(cleanText)
df = df[['cleaned_text_post', 'category']]

df.to_excel('../1-data/cleaned_test_dataset_10pc.xlsx', index=False)
