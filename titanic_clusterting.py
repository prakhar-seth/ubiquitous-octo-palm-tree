from os import error
import matplotlib.pyplot as plt
from matplotlib import style
from numpy.lib.arraysetops import unique
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import preprocessing
df=pd.read_excel('titanic.xls')
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
clf=KMeans(n_clusters=2)
clf.fit(x)
correct=0
for i in range(len(x)):
    predict_v=np.array(x[i].astype(float))
    predict_v=predict_v.reshape(-1,len(predict_v))
    prediction=clf.predict(predict_v)
    if prediction[0]==y[i]:
        correct+=1
print (correct/len(x))