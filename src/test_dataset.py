import pandas as pd

# For the test phase, we use the "real" estimated ratio (based on Hajarian & Khanbabaloo results) on a 20000 sample size
incel_test = pd.read_csv('data/incels/incels_data_test.csv')
neutral_test = pd.read_csv('data/neutrals/neutrals_data_test.csv')

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
testing_dataset.to_csv(f'data/test_dataset_{int(ratio*100)}.csv', index=False)