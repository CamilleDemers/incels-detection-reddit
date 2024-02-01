import pandas as pd
import os
import re

""" 
This script ... [to complete]
"""

data_per_year_neutral = {
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

incels_subreddits = pd.read_csv('./utils/incels_subreddits.csv')['subreddit'].tolist()
incels_subreddits += [x[:-16] for x in os.listdir('../1-data/incels/subreddits/')]
incels_subreddits = list(set(incels_subreddits))

def clean_filter_chunk(chunk, incels_subreddits):
    chunk['date_post'] = pd.to_datetime(chunk['created_utc'], unit='s').dt.year
    chunk = chunk[['date_post', 'id', 'author', 'subreddit', 'body']]
    chunk = chunk = chunk.rename(
        columns={
            'body': 'text_post',
            'created_utc': 'date_post',
            'id': 'id_post'}
        )
    
    # Remove data that come from incels subreddits
    chunk = chunk[~chunk['subreddit'].isin(incels_subreddits)]

    # Remove URLs
    chunk['text_post'] = chunk['text_post'].apply(lambda x: re.sub(r'http\S+', '', x))

    # Remove empty posts
    chunk = chunk[(chunk['text_post'] != '[removed]') & (chunk['text_post'] != '[deleted]')]

    # Remove bot actions
    chunk = chunk[chunk['author'] != 'AutoModerator']

    # Remove na values 
    chunk = chunk.dropna()
    chunk = chunk[~(chunk['text_post'].isna())]

    return chunk

sample_train_neutrals = pd.DataFrame()
sample_test_neutrals = pd.DataFrame()

years = os.listdir('../1-data/neutral/archives_pushshift/')
for year in years:
    try:
        chunks = pd.read_json(
            f'../1-data/neutral/archives_pushshift/{year}',
            compression=dict(method='zstd', max_window_size=2147483648), 
            lines=True, 
            chunksize=50000
        )

        year = int(year[3:-7])
        nb_posts_train = data_per_year_neutral[year]
        nb_posts_test = data_per_year_neutral[year]
        nb_posts_total =  nb_posts_train + nb_posts_test 

        print("On essaie pour l'année", year)

        nb_chunk_traites = 0
        sample_neutral_year = pd.DataFrame()

        for chunk in chunks:
            data = clean_filter_chunk(chunk, incels_subreddits)
            sample_neutral_year = pd.concat([sample_neutral_year, data])

            nb_posts = len(sample_neutral_year)

            nb_chunk_traites += 1
            print(nb_posts, '('+str(nb_chunk_traites)+' chunks traités)')
            if (nb_posts >= nb_posts_total):
                sample_neutral_year = sample_neutral_year.sample(nb_posts_total)
                sample_neutral_year_train = sample_neutral_year.sample(nb_posts_train)
                sample_neutral_year_test = sample_neutral_year.drop(index=sample_neutral_year_train.index)

                sample_train_neutrals = pd.concat([sample_train_neutrals, sample_neutral_year_train])
                sample_test_neutrals = pd.concat([sample_test_neutrals, sample_neutral_year_test])

                break

    except Exception as e:
        print(year, e)    

sample_train_neutrals['category'] = 'neutral'
sample_test_neutrals['category'] = 'neutral'


sample_train_neutrals.to_csv('../1-data/neutral/neutrals_data_training.csv', index=False)
sample_test_neutrals.to_csv('../1-data/neutral/neutrals_data_test.csv', index=False)