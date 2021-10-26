import numpy as np
from sklearn import model_selection
data=np.loadtxt("data.csv",delimiter=",")
x=data[:,0]
y=data[:,1]
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2)

def fit(x_train, y_train):
    num=(x_train*y_train).mean()-x_train.mean()*y_train.mean()
    den=(x_train**2).mean()-x_train.mean()**2
    m=num/den
    c=y_train.mean()-m*x_train.mean()
    return m,c

def predict(x,m,c):
    return m*x+c

def score(y_truth,y_predicted):
    u=((y_truth-y_predicted)**2).sum()
    v=((y_truth-y_truth.mean())**2).sum()
    return 1-(u/v)
def cost(x,y,m,c):
    return ((y-m*x+c)**2).mean()

m,c=fit(x_train,y_train)
y_test_pred=predict(x_test,m,c)
print("Test score:",(score(y_test,y_test_pred)))

y_train_pred=predict(x_train,m,c)
print("train score:",(score(y_train,y_train_pred)))
print("M,c",m,c)
print("Cost on trainin data",cost(x_train,y_train,m,c)/80)
