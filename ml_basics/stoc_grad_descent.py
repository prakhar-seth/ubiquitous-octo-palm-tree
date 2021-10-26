import numpy as np
def step_gradient(points,learning_rate,m,c):
    new_m=m
    new_c=c
    n=len(points)
    for i in range(0,n):
        x=points[i,0]
        y=points[i,1]
        new_m=new_m- learning_rate* (-2/n)*(y-new_m*x-c)*x
        new_c=new_c-learning_rate* (-2/n)*(y-new_m*x-c)
    return new_m,new_c

def gd(points,learning_rate,num_iteration):
    m=0
    c=0
    for i in range(num_iteration):
        m,c=step_gradient(points,learning_rate,m,c)
        print(i,"cost:",cost(points,m,c))
    return m,c

def cost(points,m,c):  
    total_cost=0
    n=len(points)
    for i in range(0,n):
        x=points[i,0]
        y=points[i,0]
        total_cost+=(1/n)*((y-m*x-c)**2)
    return total_cost

data=np.loadtxt("data.csv",delimiter=",")
learning_rate=0.0001
num_iteration=100
m,c=gd(data,learning_rate,num_iteration)
print (m,c)
