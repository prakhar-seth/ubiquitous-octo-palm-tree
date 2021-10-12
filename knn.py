import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import neighbors
import pandas as pd

df=pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999,True)
df.drop(['id'],1,inplace=True)
x=np.array(df.drop(['class'],1))
y=np.array(df['class'])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)
eg=np.array([4,3,2,1,1,3,2,3,1,])
eg=eg.reshape(1,-1)
pred=clf.predict(eg)
print(pred)