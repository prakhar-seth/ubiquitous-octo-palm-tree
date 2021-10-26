from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as style
import random

#style.use('fivethirtyeight')
#xs=np.array([1,2,3,4,5,6], dtype=np.float64)
#ys=np.array([5,4,6,5,6,7], dtype=np.float64)
def create_dataset(hm, varaince, step=2, correlation=False):
    val=1
    ys=[]
    for i in range(hm):
        y=val+ random.randrange(-varaince,varaince)
        ys.append(y)
        if correlation and correlation=='pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-= step
    xs=[i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64),np.array(ys, dtype=np.float64)
def best_fit_slope_and_intecept(xs,ys):
    m=(mean(xs)*mean(ys)-mean(xs*ys))/((mean(xs)**2)-mean(xs**2))
    b=mean(ys)-m*mean(xs)
    return m,b
def squared_error(ys_origin, ys_line):
    return sum((ys_line-ys_origin)**2)
def coeg_of_determination(ys_origin,ys_line):
    y_mean=[mean(ys_origin) for  y in ys_origin]
    squared_error_reg=squared_error(ys_origin,ys_line)
    squared_error_y_mean=squared_error(ys_origin, y_mean)
    return 1-(squared_error_reg/squared_error_y_mean)

xs,ys=create_dataset(40,10,2,correlation='pos')
m,b=best_fit_slope_and_intecept(xs,ys)
print(m) 
reg_line=[(m*x+b) for x in xs]
r_squared=coeg_of_determination(ys,reg_line)
print(r_squared)
plt.scatter(xs,ys)
plt.plot(xs,reg_line)
plt.show()