import os
import pandas as pd 

folder = 'results'
files = [os.path.join(folder, file) for file in os.listdir(folder) if 'sbert_results_train' in file]

df = pd.concat([pd.read_csv(file) for file in files])
df.sort_values(
    by='rank_test_f1_macro', ascending=False
    ).to_csv('results/results_training_sbert.csv', index=False)