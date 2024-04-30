import pandas as pd

# neutral_train = pd.read_csv('../1-data/neutral/neutrals_data_training.csv') # Size should be 100 000 
# incel_train = pd.read_csv('../1-data/incels/incels_data_training.csv') # Size should be 100 000

# ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# sample_size = 40000

# for ratio in ratios:
#     incel_data = incel_train.sample(n=int(ratio*sample_size))
#     print('-----')
#     print('incels', len(incel_data))
#     neutral_data = neutral_train.sample(n=int((1-ratio)*sample_size))
#     print('neutrals', len(neutral_data))

#     dataset = pd.concat([incel_data, neutral_data])
#     print('total', len(dataset))
#     print('-----')

#     dataset.to_excel(f'../1-data/training_datasets/train_dataset_{int(ratio*100)}pc.xlsx', index=False, engine='xlsxwriter')

# For the test phase, we use the "real" estimated ratio on a 20 000 sample size
sample_size = 20000
ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for ratio in ratios:
    incel_test = pd.read_csv('../1-data/incels/incels_data_test.csv').sample(n=int(ratio*sample_size))
    incel_test['category'] = 'incel'
    neutral_test = pd.read_csv('../1-data/neutral/neutrals_data_test.csv').sample(n=(1-ratio)*sample_size) 


    testing_dataset = pd.concat([neutral_test, incel_test])
    testing_dataset.to_excel(f'../1-data/test_dataset_{int(ratio*100)}.xlsx', index=False, engine='xlsxwriter')