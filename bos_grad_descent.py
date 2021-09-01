import numpy as np
from sklearn import datasets

from sklearn import preprocessing
def step_grad(X, Y, m, learning_rate):
    m_slope=np.zeros(len(X[0]))
    for i in range(len(X)):
        x=X[i]
        y=Y[i]
        for j in range(len(x)):
            m_slope[j]+=(-2/len(X))*(y-sum(m*x))*x[j]
    new_m=m-(learning_rate*m_slope)
    return new_m

def cost(m, x, y):
    cost=0
    for i in range(len(x)):
        cost+=(1/len(x))*((y[i]-sum(m*x[i]))**2)
    print(cost)

def gd(x, y, learning_rate, iterations):
    m=np.zeros(len(x[0]))
    for i in range(iterations):
        m=step_grad(x, y, m, learning_rate)
        print("itr= ", i, "cost=", end=' ')
        cost(m, x, y)
    return m

def gradient_descent(x, y):
    iterations=300
    learning_rate=0.1
    x=np.append(x, np.ones(len(x)).reshape(-1, 1), axis=1)
    m=gd(x, y, learning_rate, iterations)
    return m
boston =datasets.load_boston()
x=boston.data
y=boston.target
scaler=preprocessing.StandardScaler()
scaler.fit(x)
x=scaler.transform(x)
m=gradient_descent(x, y)
