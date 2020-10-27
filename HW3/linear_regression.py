
# coding: utf-8

# In[28]:


import numpy as np
import matplotlib.pyplot as plt
import copy
import math


# In[87]:


def inverse(A):
    n = len(A)
    A_copy = copy.deepcopy(A)
    I = [[0 for i in range(n)] for j in range(n)]
    for i in  range(n):
        I[i][i] = 1.0
    I_copy = copy.deepcopy(I)
    indices = list(range(n))
    for fd in range(n):
        fdScalar = 1.0/A_copy[fd][fd]
        for j in range(n):
            A_copy[fd][j] *= fdScalar
            I_copy[fd][j] *= fdScalar
        for i in indices[0:fd]+indices[fd+1:]:
            crScaler = A_copy[i][fd]
            for j in range(n):
                A_copy[i][j] = A_copy[i][j] - crScaler * A_copy[fd][j]
                I_copy[i][j] = I_copy[i][j] - crScaler * I_copy[fd][j]
    return I_copy


# In[88]:


def transpose(A):
    AT = []
    for i in range(len(A[0])):
        line = []
        for j in range(len(A)):
            line.append(A[j][i])
        AT.append(line)
    return AT


# In[89]:


def dot(A,B):
    C = []
    for i in range(len(A)):
        line = []
        for j in range(len(B[0])):
            temp = 0
            for k in range(len(A[0])):
                temp += A[i][k]*B[k][j]
            line.append(temp)
        C.append(line)
    return C


# In[90]:


data = []
file = open('linear_data.txt')
for line in file:
    data.append(line)
dim = len(data)


# In[91]:


x_plt = []
X = []
y = []
y_plt = []
for line in data:
    x_plt.append(float(line.split(',')[0]))
    temp = []
    temp.append(float(line.split(',')[1]))
    y_plt.append(float(line.split(',')[1]))
    y.append(temp)
    d = []
    for i in range(0, dim):
        d.append(float(line.split(',')[0])**i)
    X.append(d)


# In[92]:


Xt = transpose(X[:])
XtX = dot(Xt[:], X[:])
XtX_temp = copy.deepcopy(XtX)
XtX_inv = inverse(XtX_temp)
XtX_invXt = dot(XtX_inv[:], Xt[:])
w = dot(XtX_invXt[:], y[:])


# In[93]:


def PolyCoefficients(x, coeffs):
    fx = []
    for i in range(len(x)):
        temp = 0
        for j in range(len(coeffs)):
            temp += coeffs[j]*(x[i]**j)
        fx.append(temp)
    return fx


# In[94]:


def error(x, y, coeffs):
    err = 0.0
    for i in range(len(x)):
        temp = 0
        for j in range(len(coeffs)):
            temp += coeffs[j]*(x[i]**j)
        err += abs(temp-y[i])
    return err


# In[96]:


w_plt = []
for i in range(len(w)):
    w_plt.append(w[i][0])

w_copy = w_plt[:]
err = []
for i in range(len(w_plt)):
    #print(w_copy)
    w_copy[len(w_plt)-i-1] = 0
    err.append(error(x_plt, y_plt, w_copy))

min_err = min(err)
for i in range(err.index(min_err)+1):
    w_plt[len(w_plt)-i-1] = 0

x = np.linspace(min(x_plt), max(x_plt), 100)
plt.plot(x, PolyCoefficients(x, w_plt))
plt.scatter(x_plt, y_plt)
plt.show()
equation = ""
idx = 0
for idx in range(len(w_plt)):
    if w_plt[len(w_plt)-idx-1] != 0:
        if idx < len(w_plt)-1:
            item = str(w_plt[len(w_plt)-idx-1])+"X^"+str(len(w_plt)-idx-1)
        else:
            item = str(w_plt[len(w_plt)-idx-1])
        if w_plt[len(w_plt)-idx-1] >= 0:
	        item = "+"+item
        equation+=item
print("Fitting Line: ", equation)
print("Total Error: ", error(x_plt, y_plt, w_plt))

