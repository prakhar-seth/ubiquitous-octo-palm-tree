from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
df=pd.read_csv("train_log resg.csv")
df.drop('Name',axis=1,inplace=True)
df.drop('Cabin',axis=1,inplace=True)
df.drop('Fare',axis=1,inplace=True)
df['sex']=df['Sex'].map({"female":0, "male":1})
df.drop('Sex',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df['embarked']=df.Embarked.map({"S":0, "C":0.5, "Q":1})
df.drop('Embarked', inplace=True, axis=1)
df.fillna(0,inplace=True)
y=df['Survived']
df.drop('Survived',axis=1,inplace=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(df, y)
clf1=LogisticRegression()
clf1.fit(x_train,y_train)
print(clf1.score(x_test,y_test))
