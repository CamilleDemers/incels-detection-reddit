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

incels_subreddits = pd.read_csv('./utils/subreddit_descriptions.csv')['Subreddit'].tolist()
incels_subreddits = [x[3:] for x in incels_subreddits]

def clean_filter_chunk(chunk, incels_subreddits):
    chunk = chunk.rename(
        columns={
            'selftext': 'text_post',
            'body': 'text_post',
            'created_utc': 'date_post',
            'id': 'id_post'}
        )
    
    chunk['date_post'] = pd.to_numeric(chunk['date_post'], errors='coerce')
    chunk['date_post'] = pd.to_datetime(chunk['date_post'], unit='s').dt.year 
    
    chunk = chunk[['date_post', 'id_post', 'author', 'subreddit', 'text_post']]

    # Remove na values 
    chunk = chunk.dropna()
    
    # Remove data that come from incels subreddits
    chunk = chunk[~(chunk['subreddit'].astype(str).isin(incels_subreddits))]
    chunk = chunk[~(chunk['subreddit'].str.contains('incel', na=False))]

    # Remove URLs
    chunk['text_post'] = chunk['text_post'].str.replace(r'http\S+', ' ', regex=True)

    # Remove new line delimiters
    chunk['text_post'] = chunk['text_post'].str.replace('\n', ' ')

    # Remove HTML entities
    html_entity_pattern = re.compile(r'&[a-zA-Z0-9#]+;')
    chunk['text_post'] = chunk['text_post'].str.replace(html_entity_pattern, ' ', regex=True)

    # Remove empty posts
    chunk = chunk[(chunk['text_post'] != '[removed]') & (chunk['text_post'] != '[deleted]')]
    chunk = chunk[chunk['text_post'] != '']
    chunk = chunk[chunk['text_post'].str.len() > 1]  # 1 character only

    # Remove bot actions
    chunk = chunk[chunk['author'] != 'AutoModerator']
    
    # Lowercasing 
    chunk['text_post'] = chunk['text_post'].astype(str).apply(lambda x: x.lower())

    # Remove duplicates (if any)
    chunk = chunk.drop_duplicates(subset='id_post')

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

folder = '../1-data/neutrals/the-eye_pushshift'

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

                sample_neutral_year_test = sample_neutral_year.drop(index=sample_neutral_year_train.index)

                sample_train_neutrals = pd.concat([sample_train_neutrals, sample_neutral_year_train])
                sample_test_neutrals = pd.concat([sample_test_neutrals, sample_neutral_year_test])

                break

    except Exception as e:
        print(file, e)

sample_train_neutrals['category'] = 'neutral'
sample_test_neutrals['category'] = 'neutral'

sample_train_neutrals.to_csv('../1-data/neutrals/neutrals_data_training.csv', index=False)
sample_test_neutrals.to_csv('../1-data/neutrals/neutrals_data_test.csv', index=False)
