import pandas as pd
import os
import re
import io
import zstandard as zstd
import json

"""
This script reads the zst files gathered from The-Eye-Eu (PushShift Archive). It extracts the
data from non-Incels subreddits, does some cleaning and filtering, dumps the data into a Pandas DataFrame and exports 
train and test samples of it to csv files. 

The script is adapted from the example scripts provided by Watchful1 @ The-Eye (https://github.com/Watchful1/PushshiftDumps)
"""

incels_subreddits = pd.read_csv('src/utils/subreddit_descriptions.csv')['Subreddit'].tolist()
incels_subreddits = [x[3:] for x in incels_subreddits]

def clean_filter_chunk(chunk, incels_subreddits):
    chunk = chunk.rename(
        columns={
            'selftext': 'text_post',
            'body': 'text_post',
            'created_utc': 'date_post',
            'id': 'id_post'}
        )
    
    # Conversion et sélection des colonnes
    chunk['date_post'] = pd.to_numeric(chunk['date_post'], errors='coerce')
    chunk['date_post'] = pd.to_datetime(chunk['date_post'], unit='s', errors='coerce').dt.year
    chunk = chunk[['date_post', 'id_post', 'author', 'subreddit', 'text_post']]

    # Nettoyage et filtrage
    chunk = (
        chunk
        .dropna()  # Suppression des valeurs manquantes
        .query("author != 'AutoModerator'")  # Suppression des actions de bot
    )

    # Filtrer les subreddits liés aux "incels"
    chunk = chunk[
        ~chunk['subreddit'].astype(str).isin(incels_subreddits) & 
        ~chunk['subreddit'].str.contains('incel', na=False)
    ]

    # Nettoyage du texte
    chunk['text_post'] = (
        chunk['text_post']
        .astype(str)
        .str.replace(r'http\S+', ' ', regex=True)  # Suppression des URLs
        .str.replace('\n', ' ')                   # Suppression des sauts de ligne
        .str.replace(r'&[a-zA-Z0-9#]+;', ' ', regex=True)  # Suppression des entités HTML
        .str.strip()                              # Suppression des espaces inutiles
        .str.replace(r'\s\s+', ' ', regex=True)   # Suppression des espaces multiples
        .str.lower()                              # Mise en minuscules
    )

    # Filtrer les posts invalides ou vides
    chunk = chunk[
        (chunk['text_post'] != '[removed]') &
        (chunk['text_post'] != '[deleted]') &
        (chunk['text_post'] != '') &
        (chunk['text_post'].str.len() > 1)  # Minimum 2 caractères
    ]

    # Suppression des doublons
    chunk = chunk.drop_duplicates(subset='id_post')
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
            print(chunk.columns)
            yield pd.DataFrame(chunk)
            
sample_train_neutrals = pd.DataFrame()
sample_test_neutrals = pd.DataFrame()

folder = 'data/neutrals/the-eye_pushshift'

for file in os.listdir(folder):
    try:
        sub_or_comment = 'submission' if 'RS' in file else 'comment'
        filepath = f'{folder}/{file}'
        chunks = read_zst_file(filepath)
        
        year_int = int(file[3:-7])
        nb_posts_per_year_train = 5000 
        nb_posts_per_year_test = 2250 #
        nb_posts_total = nb_posts_per_year_train + nb_posts_per_year_test

        print("Processing year", year_int)

        nb_chunk_traites = 0
        sample_neutral_year = pd.DataFrame()

        for chunk in chunks:
            data = clean_filter_chunk(chunk, incels_subreddits)
            data['pub_type'] = sub_or_comment
            sample_neutral_year = pd.concat([sample_neutral_year, data])

            nb_posts = len(sample_neutral_year)

            nb_chunk_traites += 1
            print(nb_posts, f'({nb_chunk_traites} chunks processed)')
            if nb_posts >= nb_posts_total:
                sample_neutral_year = sample_neutral_year.sample(nb_posts_total)
                sample_neutral_year_train = sample_neutral_year.sample(nb_posts_per_year_train)


                # Cette ligne sert à vérifier qu'aucune donnée se trouvant dans le jeu de données d'apprentissage
                # ne se retrouve dans le jeu de données test
                sample_neutral_year_test = sample_neutral_year.drop(index=sample_neutral_year_train.index)

                sample_train_neutrals = pd.concat([sample_train_neutrals, sample_neutral_year_train])
                sample_test_neutrals = pd.concat([sample_test_neutrals, sample_neutral_year_test])

                break

    except Exception as e:
        print(file, e)

sample_train_neutrals['category'] = 'neutral'
sample_test_neutrals['category'] = 'neutral'

sample_train_neutrals.to_csv('data/neutrals/neutrals_data_training.csv', index=False)
sample_test_neutrals.to_csv('data/neutrals/neutrals_data_test.csv', index=False)
