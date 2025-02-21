import pandas as pd
import re
import os
import io
import zstandard as zstd
import json

""" 
This script reads data (zst compressed archives) from specific subreddits gathered by The-Eye-Eu (PushShift Archive). It extracts the
data from Incels subreddits (according to the categorization made by Ribeiro et al., 2021), does some cleaning and filtering, dumps the data into a Pandas DataFrame and exports 
a sample of it to a csv file. 

The script is adapted from the example scripts provided by Watchful1 @ The-Eye (https://github.com/Watchful1/PushshiftDumps)
"""

def clean_filter_subreddit_chunk_incels(chunk):
    chunk = chunk.rename(
        columns={
            'selftext': 'text_post',
            'body': 'text_post',
            'created_utc': 'date_post',
            'id': 'id_post'}
        )
    
    chunk['date_post'] = pd.to_numeric(chunk['date_post'], errors='coerce')
    chunk['date_post'] = pd.to_datetime(chunk['date_post'], unit='s', errors='coerce').dt.year
    chunk = chunk[['id_post', 'date_post', 'subreddit', 'author', 'text_post']]

    # Filtrer les posts après 2014
    chunk = chunk[chunk['date_post'] >= 2014]

    # Nettoyage et transformations
    chunk['text_post'] = (
        chunk['text_post']
        .astype(str) # convertir en str
        .str.strip() # enlever les espaces au début et à la fin 
        .str.replace(r'http\S+', ' ', regex=True) # enlever les urls
        .str.replace('\n', ' ') # remplacer les sauts de ligne par des espaces
        .str.replace(r'&[a-zA-Z0-9#]+;', ' ', regex=True) # enlever les entités HTML
        .str.replace(r'\s\s+', ' ', regex=True) # enlever les espaces superflues
        .str.lower()
    )

    # Filtrer les posts invalides (vides, supprimés, actions de bots, etc.)
    chunk = chunk[
        (chunk['text_post'] != '[removed]') &
        (chunk['text_post'] != '[deleted]') &
        (chunk['text_post'] != '') &
        (chunk['text_post'] != '  ') &
        (chunk['text_post'].str.len() > 1) &
        (chunk['author'] != 'AutoModerator')
    ]

    # Supprimer les valeurs manquantes
    chunk = chunk.dropna()

    # Remove duplicates (if any)
    chunk = chunk.drop_duplicates('id_post')
    chunk = chunk.drop_duplicates(subset=['text_post', 'subreddit']) 
    
    return chunk

def read_zst_file(filepath, chunk_size=50000):
    dctx = zstd.ZstdDecompressor(
        max_window_size=2147483648
    )
    with open(filepath, 'rb') as f:
        stream_reader = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
        chunk = []
        for line in text_stream:
            chunk.append(json.loads(line))
            if len(chunk) == chunk_size:
                yield pd.DataFrame(chunk)
                chunk = []
        if chunk:
            yield pd.DataFrame(chunk)

# - Read data from specific subreddits archives (from The Eye PushShift Archive)
folder = 'data/incels/the-eye_pushshift/'

data_subreddits = pd.DataFrame()

for file in os.listdir(folder):
    try:
        filepath = f'{folder}/{file}'
        sub_or_comment = 'submission' if 'submissions' in file else 'comment'
        chunks = read_zst_file(filepath)
        data = pd.concat(
            [clean_filter_subreddit_chunk_incels(chunk) for chunk in chunks
            ]
        )

        data['pub_type'] = sub_or_comment
        data_subreddits = pd.concat([data_subreddits, data])
        print(file, '--', '✔️')

    except Exception as e:
        print(file, e) 


data_subreddits = data_subreddits[~data_subreddits['text_post'].isna()]
data_subreddits['category'] = 'incel'

# Extraire un échantillon
# Données d'apprentissage

sample_train_incels = pd.DataFrame()
sample_test_incels = pd.DataFrame()

for year in list(range(2014, 2024)):
    sample_year = data_subreddits[data_subreddits['date_post'] == year]
    sample_comments = sample_year[sample_year['pub_type'] == 'comment'].sample(n=5000)
    
    sample_submissions = sample_year[sample_year['pub_type'] == 'submission'].sample(n=5000)

    sample = pd.concat([sample_comments, sample_submissions])
    sample_train_incels = pd.concat([sample_train_incels, sample])


sample_train_incels.to_csv('data/incels/incels_data_training.csv', index=False)

# Données test
sample = pd.DataFrame()

# Cette ligne sert à vérifier qu'aucune donnée se trouvant dans le jeu de
# données d'apprentissage ne se retrouve dans le jeu de données test
data_subreddits = data_subreddits.drop(index=sample_train_incels.index)

for year in range(2014, 2024):
    sample_year = data_subreddits[data_subreddits['date_post'] == year]
    sample_comments = sample_year[sample_year['pub_type'] == 'comment'].sample(n=250)

    sample_submissions = sample_year[sample_year['pub_type'] == 'submission'].sample(n=250)

    sample = pd.concat([sample_comments, sample_submissions])
    sample_test_incels = pd.concat([sample_test_incels, sample])

sample_test_incels.to_csv('data/incels/incels_data_test.csv', index=False)