
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import style
import warnings
import random
import pandas as pd
style.use('fivethirtyeight')
def k_near_neigh(data,predict,k=3):
    if len(data)>=k:
        warnings.warn('k is less than valoe  less than total voting groups!')
    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    votes=[i[1] for i in sorted(distances)[:k]]
    #print(Counter(votes).most_common(1))
    votes_result=Counter(votes).most_common(1)[0][0]
    return votes_result
df=pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
full_data=df.astype(float).values.tolist() #to covert values to list
random.shuffle(full_data)
test_size=0.2
train_set={2:[],4:[]}
test_set={2:[],4:[]}
train_data=full_data[:-int(test_size*len(full_data))]
test_data=full_data[-int(test_size*len(full_data)):]
for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct =0
total=0
for group in test_set:
    for data in test_set[group]:
        vote=k_near_neigh(train_set,data,5)
        if group ==vote:
            correct+=1
        total+=1

print('Accuracy',correct/total)