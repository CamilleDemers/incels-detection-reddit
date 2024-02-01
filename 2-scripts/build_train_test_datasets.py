import pandas as pd

neutral_train = pd.read_csv('../1-data/neutral/neutrals_data_training.csv') # Size should be 100 000 
incel_train = pd.read_csv('../1-data/incels/incels_data_training.csv') # Size should be 100 000

ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
sample_size = 50000

for ratio in ratios:
    incel_data = incel_train.sample(n=int(ratio*sample_size))
    print('-----')
    print('incels', len(incel_data))
    neutral_data = neutral_train.sample(n=int((1-ratio)*sample_size))
    print('neutrals', len(neutral_data))

    dataset = pd.concat([incel_data, neutral_data])
    print('total', len(dataset))
    print('-----')

    dataset.to_excel(f'../1-data/training_datasets/train_dataset_{int(ratio*100)}pc.xlsx', index=False, engine='xlsxwriter')

# For the test phase, we use the "real" estimated ratio on a 10 000 sample size
neutral_test = pd.read_csv('../1-data/neutral/neutrals_data_test.csv').sample(n=9000) 
incel_test = pd.read_csv('../1-data/incels/incels_data_test.csv').sample(n=1000)
incel_test['category'] = 'incel'

testing_dataset = pd.concat([neutral_test, incel_test])
testing_dataset.to_excel('../1-data/test_dataset_10pc.xlsx', index=False, engine='xlsxwriter')