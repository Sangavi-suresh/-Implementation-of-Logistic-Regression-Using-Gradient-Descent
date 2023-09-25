# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6.Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sangavi Suresh
RegisterNumber:  212222230130
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

df=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=df[:,[0,1]]
y=df[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return J,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h))) / X.shape[0]
  return J

def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y) / X.shape[0]
  return grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(X_train,y),
                        method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min() - 1,X[:,0].max()+1
  y_min,y_max=X[:,1].min() - 1,X[:,0].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),
                    np.arange(y_min,y_max,0.1))

  X_plot = np.c_[xx.ravel(),yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label='admitted')
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label='NOT admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()

plotDecisionBoundary(res.x,X,y)

prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob = sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)

np.mean(predict(res.x,X)==y)

```

## Output:

Array Value of x:

![image](https://github.com/Sangavi-suresh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541861/d119286b-e0ba-4c5f-8849-63ea88626594)

Array Value of y:

![image](https://github.com/Sangavi-suresh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541861/6185e27b-06f3-4057-b4f2-86bf61813f53)

Exam 1 - score graph:

![image](https://github.com/Sangavi-suresh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541861/d52cabd8-06e7-4818-bfff-be70d666bed4)

Sigmoid function graph:

![image](https://github.com/Sangavi-suresh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541861/511edda4-94d9-4610-a742-4132f863505a)

X_train_grad value:

![image](https://github.com/Sangavi-suresh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541861/1fd79dec-8c97-4ff9-9064-aaccc3e9eb91)

Y_train_grad value:

![image](https://github.com/Sangavi-suresh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541861/215d34c5-e3ce-467b-ae45-c7c79539ff3e)

Print res.x:

![image](https://github.com/Sangavi-suresh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541861/d16e847b-389d-41a8-ac7a-67c833dd9947)


Decision boundary - graph for exam score:

![image](https://github.com/Sangavi-suresh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541861/7657c526-980a-4c92-b33d-46ab2642425d)

Proability value:

![image](https://github.com/Sangavi-suresh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541861/32eb1d16-4cd6-4e13-ace8-d0538be161d5)

Prediction value of mean:

![image](https://github.com/Sangavi-suresh/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118541861/f81b488c-9f1a-44c6-9c41-054ee5278a9d)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

