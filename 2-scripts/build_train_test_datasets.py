import pandas as pd

neutral_train = pd.read_csv('../1-data/neutrals/neutrals_data_training.csv')  
incel_train = pd.read_csv('../1-data/incels/incels_data_training.csv') 

ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
sample_size = 40000

for ratio in ratios:
    nb_incels = int(ratio*sample_size)
    nb_neutrals = sample_size - nb_incels

    incel_data = incel_train.groupby('date_post', group_keys=False).apply(lambda x: x.sample(int(nb_incels/10)))
    print('-----')
    print('incels', nb_incels)

    neutral_data = neutral_train.groupby('date_post', group_keys=False).apply(lambda x: x.sample(int(nb_neutrals/10)))
    print('neutrals', nb_neutrals)

    dataset = pd.concat([incel_data, neutral_data])
    print('total', len(dataset))
    print('-----')

    dataset.to_excel(f'../1-data/training_datasets/train_dataset_{int(ratio*100)}pc.xlsx', index=False, engine='xlsxwriter')

# For the test phase, we use the "real" estimated ratio (based on Hajarian & Khanbabaloo results) on a 20000 sample size
incel_test = pd.read_csv('../1-data/incels/incels_data_test.csv')
neutral_test = pd.read_csv('../1-data/neutrals/neutrals_data_test.csv')

sample_size = 20000
ratio = 0.1

nb_incels = int(ratio*sample_size)
nb_neutrals = sample_size - nb_incels 


incel_test = incel_test.groupby('date_post', group_keys=False).apply(lambda x: x.sample(int(nb_incels/10)))
print('-----')
print('incels', nb_incels)

neutral_test = neutral_test.groupby('date_post', group_keys=False).apply(lambda x : x.sample(n=int(nb_neutrals/10))) 
print('-----')
print('neutrals', nb_neutrals)

testing_dataset = pd.concat([neutral_test, incel_test])
testing_dataset.to_excel(f'../1-data/test_dataset_{int(ratio*100)}.xlsx', index=False, engine='xlsxwriter')