import numpy as np
from sklearn import datasets
import pandas as pd


def step_grad(data,y,m,learning_rate):
    n=data.shape[1]
    m_slope=np.zeros(n)
    m=data.shape[0]
    for i in range(n):
        for j in range(m):
            x=data.loc[j,:]
            m_slope[i]+=(-2/n)*((y-np.dot(x,m))*x[i])
    new_m=m-learning_rate*m_slope
    return new_m

def gd(points,y,learning_rate,num_iteration):
    n=points.shape[1]
    m=np.zeros(n)
    for i in range(num_iteration):
        m=step_grad(points,y,learning_rate,m)
        print("cost of ",i,"iterartion is", cost(points,y,m))
    return m         
def cost(data,y,m):  
    total_cost=0
    n=data.shape[1]
    for i in range(0,n):
        x=data.loc[i,:].reshape(-1,1)
        total_cost+=(1/n)*(y-np.dot(m,x)**2)

    return total_cost
boston =datasets.load_boston()
x=boston.data
y=boston.target
df=pd.DataFrame(x)
df.columns=boston.feature_names
df['c']=1
m=gd(df,y,0.001,10)
print(m)