import pandas as pd
import re
import os

""" 
This script reads the ndjson file made available by Ribeiro et al. (2021) [https://zenodo.org/records/4007913]
as well as the data from specific subreddits gathered by The-Eye-Eu archive (PushShift Archive). It extracts the
data from Incels subreddits, does some cleaning and filtering, dumps the data into a Pandas DataFrame and exports 
train and test samples of it to csv files. We use a stratified sampling method to maintain proportions of
posts through the years.
"""

data_per_year_incels = {
    2014:250,
    2015:1000,
    2016:1500,
    2017:2500,
    2018:5000,
    2019:6250,
    2020:7500,
    2021:11000,
    2022:15000
}

subreddits = pd.read_csv('./utils/incels_subreddits.csv')['subreddit'].tolist()

def clean_filter_chunk_incels(chunk, subreddits):
    chunk['date_post'] = pd.to_datetime(chunk['date_post'], unit='s').dt.year
    chunk = chunk[chunk['date_post'] >= 2014]
    chunk = chunk[chunk['subreddit'].isin(subreddits)]
    chunk = chunk[['id_post', 'date_post', 'subreddit', 'author', 'text_post']]
    
    
    # Remove URLs
    chunk['text_post'] = chunk['text_post'].apply(lambda x: re.sub(r'http\S+', '', x))
    
    # Remove empty posts
    chunk = chunk[(chunk['text_post'] != '[removed]') & (chunk['text_post'] != '[deleted]')]

    # Remove bot actions
    chunk = chunk[chunk['author'] != 'AutoModerator']
    
    return chunk

def clean_filter_subreddit_chunk_incels(chunk, subreddits):
    chunk = chunk[['body', 'created_utc', 'id', 'author', 'subreddit']]
    chunk = chunk.rename(columns={'body': 'text_post', 'created_utc': 'date_post', 'id': 'id_post'})
    
    chunk['date_post'] = pd.to_datetime(chunk['date_post'], unit='s').dt.year
    chunk = chunk[['id_post', 'date_post', 'subreddit', 'author', 'text_post']]
    
    subreddit = chunk['subreddit'].unique()[0]
    
    if subreddit in subreddits:
        chunk = chunk[chunk['date_post'] >= 2020]
    else:
        chunk = chunk[chunk['date_post'] >= 2014]
    
    # Remove URLs
    chunk['text_post'] = chunk['text_post'].apply(lambda x: re.sub(r'http\S+', '', x))
    
    # Remove empty posts
    chunk = chunk[(chunk['text_post'] != '[removed]') & (chunk['text_post'] != '[deleted]')]

    # Remove bot actions
    chunk = chunk[chunk['author'] != 'AutoModerator']
    
    return chunk

# 1 - Data from Ribeiro et al. (2021)
chunks = pd.read_json('../1-data/incels/ribeiro_reddit.ndjson', lines=True, chunksize=500000)
data_ribeiro = pd.concat([clean_filter_chunk_incels(chunk, subreddits) for chunk in chunks])
print('ribeiro_reddit.ndjson', '--', '✔️')

# 2 - Specific subreddits archives
folder = '../1-data/incels/subreddits/'
files = os.listdir(folder)

data_subreddits = pd.DataFrame()

for file in files:
    data = pd.concat(
        [clean_filter_subreddit_chunk_incels(chunk, subreddits) for chunk in pd.read_json(
            f'{folder}{file}', lines=True, chunksize=500000)
        ]
    )  
    data_subreddits = pd.concat([data_subreddits, data])
    print(file, '--', '✔️')

# 2 - Combine with Ribeiro et al. dataset
df = pd.concat([data_subreddits, data_ribeiro])
df = df[~df['text_post'].isna()]

df['category'] = 'incel'

# Train / test samples
# Train
years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
sample_train_incels = pd.DataFrame()

for year in years:
    sample = df[df['date_post'] == year].sample(n=data_per_year_incels[year])
    sample_train_incels = pd.concat([sample_train_incels, sample])


sample_train_incels.to_csv('../1-data/incels/incels_data_training.csv', index=False)

# Test
df = df.drop(index=sample_train_incels.index)
sample_test_incels = pd.DataFrame()

for year in years:
    sample = df[df['date_post'] == year].sample(n=data_per_year_incels[year])
    sample_test_incels = pd.concat([sample_test_incels, sample])

sample_test_incels.to_csv('../1-data/incels/incels_data_test.csv', index=False)