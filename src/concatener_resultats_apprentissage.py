#!/usr/bin/env python
# coding: utf-8

## Ce script permet de concaténer les fichiers de résultats (csv) contenus dans le dossier results/
import os
import pandas as pd 

folder = 'results/results_training/'
files = [os.path.join(folder, file) for file in os.listdir(folder)]

df = pd.concat([pd.read_csv(file) for file in files])
df.sort_values(
    by='rank_test_f1_macro', ascending=False
    ).to_csv('results/results_training.csv', index=False)