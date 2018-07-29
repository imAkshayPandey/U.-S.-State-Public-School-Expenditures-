import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(20.0,10.0)


data=pd.read_csv('Anscombe.csv')
print(data.shape)
data=data[['education','income','young','urban']]

X=data['education'].values
Y=data['income'].values

mean_X=np.mean(X)
mean_Y=np.mean(Y)

n=len(X)

numer=0
denom=0

for i in range(n):
    numer+=(X[i]-mean_X)*(Y[i]-mean_Y) # summation(x-mean_x) (y-mean_y)/summation(x-mean_x)2
    denom=(X[i]-mean_X)**2

m=numer/denom
c=mean_Y-(m*mean_X)

ss_t=0
ss_r=0
for i in range(n):
    y_pred=m*X[i]+c
    ss_t+=(Y[i]-mean_Y)**2
    ss_r+=(Y[i]-y_pred)**2
    #pre_x+=(m*X[i]+c-Y)**2
    #pre_y+=(Y[i]-mean_Y)**2
h=1-(ss_r/ss_t)
print(h)  
    
    


print(m,c)


max_X=np.max(X)
min_X=np.min(X)

x=np.linspace(min_X,max_X,10)
y=m*x+c




plt.plot(x,y,color='#58b970',label='Regression Line')
plt.scatter(X,Y,c='#ef5423',label='Scatter Plot')

plt.xlabel('Education')
plt.ylabel('income')

plt.legend()
plt.show()


