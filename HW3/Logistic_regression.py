
# coding: utf-8

# In[140]:


import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

def normalize(X):
    x_normalized = []
    x = []
    y = []
    for item in X:
        x.append(item[0])
        y.append(item[1])
    for item in X:
        temp = []
        temp.append(-1+2*(item[0]-min(x))/(max(x)-min(x)))
        temp.append(-1+2*(item[1]-min(y))/(max(y)-min(y)))    
        x_normalized.append(temp)
    return x_normalized



def dot(w, x):
    d = 0
    for i in range(len(w)):
        d += w[i]*x[i]
    return d

def sigmoid(x):
    L_of_x = 1.0 / (1.0 + math.exp(-1.0*x))
    return L_of_x 


def M_of_x(w, x):
    z = dot(w, x)
    return sigmoid(z)


def Cost(X,y,w,m):
    sumOfErrors = 0
    for i in range(m):
        Mwxi = M_of_x(w,X[i])
        sumOfErrors += y[i]*math.log(Mwxi)+(1-y[i])*math.log(1-Mwxi)
    G_of_w = (-1.0/m) * sumOfErrors
    return G_of_w


def Cost_Derivative(X,y,w,j,m,alpha):
    sumofErrors = 0
    for i in range(m):
        xi = X[i]
        xij = xi[j]
        Mwxi = M_of_x(w,X[i])
        error = (y[i]-Mwxi)*Mwxi*(1.0-Mwxi)*X[i][j]
        sumofErrors += error
    m = len(y)
    wj = (float(alpha)) * sumofErrors
    return wj


def Gradient(X,y,w,m,alpha):
    new_w = []
    constant = alpha/m
    for j in range(len(w)):
        derivative = Cost_Derivative(X,y,w,j,m,alpha)
        temp = w[j] + derivative
        new_w.append(temp)
    return new_w

def Logistic_Regression(X,y,alpha,w,times):
    m = len(y)
    for i in range(times):
        new_w = Gradient(X,y,w,m,alpha)
        w = new_w
    return w


def calaculate(data0, data1):
    X = []
    y = []
    x_plt_0 = []
    x_plt_1 = []
    y_plt_0 = []
    y_plt_1 = []

    for line in data0:
        temp = []
        temp.append(float(line.split(',')[0]))
        temp.append(float(line.split(',')[1]))
        X.append(temp)
        x_plt_0.append(float(line.split(',')[0]))
        y_plt_0.append(float(line.split(',')[1]))
    for line in data1:
        temp = []
        temp.append(float(line.split(',')[0]))
        temp.append(float(line.split(',')[1]))
        X.append(temp)
        x_plt_1.append(float(line.split(',')[0]))
        y_plt_1.append(float(line.split(',')[1]))
    for i in range(len(data0)):
        y.append(0)
    for i in range(len(data1)):
        y.append(1)

    X_norm = normalize(X)
    for i in X_norm:
        i.append(1)
    initial_w = [0,0,0]
    alpha = 0.1
    times= 10000
    w = Logistic_Regression(X_norm,y,alpha,initial_w,times)


    #plt.scatter(x_plt_0, y_plt_0)
    #plt.scatter(x_plt_1, y_plt_1)
    #plt.show()
    
    x_norm_0 = []
    x_norm_1 = []
    y_norm_0 = []
    y_norm_1 = []
    idx = 0
    for idx in range(len(X_norm)):
        if idx < 50:
            x_norm_0.append(X_norm[idx][0])
            y_norm_0.append(X_norm[idx][1])
        else:
            x_norm_1.append(X_norm[idx][0])
            y_norm_1.append(X_norm[idx][1])

    x_norm_0 = []
    x_norm_1 = []
    y_norm_0 = []
    y_norm_1 = []
    idx = 0
    for idx in range(len(X_norm)):
        if idx < 50:
            x_norm_0.append(X_norm[idx][0])
            y_norm_0.append(X_norm[idx][1])
        else:
            x_norm_1.append(X_norm[idx][0])
            y_norm_1.append(X_norm[idx][1])
            
    x_norm = []
    for i in x_norm_0:
        x_norm.append(i)
    for i in x_norm_1:
        x_norm.append(i)
    y_norm = []
    for i in y_norm_0:
        y_norm.append(i)
    for i in y_norm_1:
        y_norm.append(i)
    idx = 0

    predict_x_0 = []
    predict_x_1 = []
    predict_y_0 = []
    predict_y_1 = []

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    print('w: ', w)
    a = -w[0]/w[1]
    b = -w[2]/w[1]
    for idx in range(len(x_norm)):
        if x_norm[idx]*float(a)-float(b) >= y_norm[idx]:
            predict_x_0.append(X[idx][0])
            predict_y_0.append(X[idx][1])
            if y[idx] == 0:
                TP += 1
            else:
                FP += 1
            
        else:
            predict_x_1.append(X[idx][0])
            predict_y_1.append(X[idx][1])
            if y[idx] == 1:
                TN += 1
            else:
                FN += 1
    plt.scatter(predict_x_0, predict_y_0)
    plt.scatter(predict_x_1, predict_y_1)
    plt.show()
    confusion_mat = {'Is Cluster 1': [TP, FN],
                     'Is Cluster 2': [FP, TN]}
    matrix = pd.DataFrame(confusion_mat)
    matrix.index = ['Predixt Cluster 1', 'Predict Cluster 2']
    print('Confusion Matrix:')
    print(matrix)
    print('Precision: ', TP/(TP+FP))
    print('Recall: ', TP/(TP+FN))

file0 = open('Logistic_data1-1.txt')
data0 = []
for line in file0:
    data0.append(line)
file0.close()
file1 = open('Logistic_data1-2.txt')
data1 = []
for line in file1:
    data1.append(line)
file1.close()
calaculate(data0, data1)

file0 = open('Logistic_data2-1.txt')
data0 = []
for line in file0:
    data0.append(line)
file0.close()
file1 = open('Logistic_data2-2.txt')
data1 = []
for line in file1:
    data1.append(line)
file1.close()
calaculate(data0, data1)