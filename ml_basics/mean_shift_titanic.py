from os import error
import matplotlib.pyplot as plt
from matplotlib import style
from numpy.lib.arraysetops import unique
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
import pandas as pd
from sklearn import preprocessing
df=pd.read_excel('titanic.xls')
original_df=pd.DataFrame.copy(df)
df.drop(['body','name'],1,inplace=True)
df.apply(pd.to_numeric,errors='ignore')

df.fillna(0,inplace=True)


def handle_non_numeric(df):
    columns=df.columns.values
    for column in columns:
        text_digits_vals={}
        def convert_to_int(val):
            return text_digits_vals[val]
        if df[column].dtype !=np.int64 and df[column].dtype!=np.float64:
            column_contents=df[column].values.tolist()
            unique_elements=set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digits_vals:
                    text_digits_vals[unique]=x
                    x+=1
            df[column]=list(map(convert_to_int,df[column]))
    return df

df=handle_non_numeric(df)
#print(df.head())
x=np.array(df.drop(['survived'],1).astype(float))
x=np.array(df.drop(['ticket'],1).astype(float))
x=preprocessing.scale(x)
y=np.array(df['survived'])
clf=MeanShift()
clf.fit(x)
labels=clf.labels_
cluster_centers=clf.cluster_centers_
original_df['cluster_group']=np.nan
for i in range(len(x)):
    original_df['cluster_group'].iloc[i]=labels[i]
n_clusters_=len(np.unique(labels))
survival_rates={}
for i in range(n_clusters_):
    temp_df=original_df[(original_df['cluster_group']==float(i))]
    survival_cluster=temp_df[(temp_df['survived']==1)]
    survival_rate=len(survival_cluster)/len(temp_df)
    survival_rates[i]=survival_rate
print(survival_rates)