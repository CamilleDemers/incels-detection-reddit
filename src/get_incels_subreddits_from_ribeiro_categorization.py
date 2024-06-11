import pandas as pd

## Read Ribeiro et al. (2021) subreddit categories file to get the list of Incels subreddits

incels_subreddits = pd.read_csv('utils/subreddit_descriptions.csv')[['Subreddit', 'Category after majority agreement']]
incels_subreddits = incels_subreddits.rename(
    columns={
        'Subreddit': 'subreddit', 
        'Category after majority agreement': 'category'
        }
    )

incels_subreddits = incels_subreddits[incels_subreddits['category'] == 'Incels']
incels_subreddits.to_csv('utils/incels_subreddits.csv', index=False)