import os
import pandas as pd 

folder = '../3-results'
files = [os.path.join(folder, file) for file in os.listdir(folder) if 'training' in file]

df = pd.concat([pd.read_csv(file) for file in files])
df.sort_values(
    by='f1-score', ascending=False
    ).to_csv('../3-results/results_training.csv', index=False)

top_20 = df.sort_values(by='f1-score', ascending=False)[:20]
df.sort_values(by='f1-score', ascending=False)