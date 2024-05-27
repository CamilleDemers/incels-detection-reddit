import pandas as pd
import re
import os
import io
import zstandard as zstd
import json

""" 
This script reads data (zst compressed archives) from specific subreddits gathered by The-Eye-Eu (PushShift Archive). It extracts the
data from Incels subreddits (according to the categorization made by Ribeiro et al., 2021), does some cleaning and filtering, dumps the data into a Pandas DataFrame and exports 
train and test samples of it to csv files. 
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
    chunk['date_post'] = pd.to_datetime(chunk['date_post'], unit='s').dt.year
    chunk = chunk[['id_post', 'date_post', 'subreddit', 'author', 'text_post']]
    
    chunk = chunk[chunk['date_post'] >= 2014]

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
    chunk = chunk[chunk['text_post'].str.len() >1] # 1 character only

    # Remove bot actions
    chunk = chunk[chunk['author'] != 'AutoModerator']

    # Remove na values
    chunk = chunk.dropna()
    chunk = chunk[~chunk['text_post'].isna()]
    
    # Lowercasing 
    chunk['text_post'] = chunk['text_post'].astype(str).apply(lambda x : x.lower())

    # Remove duplicates (if any)
    chunk = chunk.drop_duplicates('id_post')
    chunk = chunk.drop_duplicates()
    
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
folder = '../1-data/incels/the-eye_pushshift/'

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

# Train / test samples
# Train

sample_train_incels = pd.DataFrame()
sample_test_incels = pd.DataFrame()

for year in list(range(2014, 2024)):
    sample_year = data_subreddits[data_subreddits['date_post'] == year]
    sample_comments = sample_year[sample_year['pub_type'] == 'comment'].sample(n=5000)
    
    sample_submissions = sample_year[sample_year['pub_type'] == 'submission'].sample(n=5000)

    sample = pd.concat([sample_comments, sample_submissions])
    sample_train_incels = pd.concat([sample_train_incels, sample])


sample_train_incels.to_csv('../1-data/incels/incels_data_training.csv', index=False)

# Test
sample = pd.DataFrame()
data_subreddits = data_subreddits.drop(index=sample_train_incels.index)

for year in range(2014, 2024):
    sample_year = data_subreddits[data_subreddits['date_post'] == year]
    sample_comments = sample_year[sample_year['pub_type'] == 'comment'].sample(n=250)

    sample_submissions = sample_year[sample_year['pub_type'] == 'submission'].sample(n=250)

    sample = pd.concat([sample_comments, sample_submissions])
    sample_test_incels = pd.concat([sample_test_incels, sample])

sample_test_incels.to_csv('../1-data/incels/incels_data_test.csv', index=False)