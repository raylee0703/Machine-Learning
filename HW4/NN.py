
# coding: utf-8

# In[252]:


import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(0)

# In[254]:


data = []
file = open('data.txt')
groud_truth_1 = []
groud_truth_0 = []
for line in file:
    xy = []
    target = []
    temp = []
    xy.append(float(line.split(',')[0]))
    xy.append(float(line.split(',')[1]))
    target.append(int(line.split(',')[2]))
    if int(line.split(',')[2]) == 1:
        groud_truth_1.append(xy)
    else:
        groud_truth_0.append(xy)
    temp.append(xy)
    temp.append(target)
    data.append(temp)


# In[255]:


n_input = 2+1
n_hidden = 2
n_output = 1
weight_i = []
for i in range(n_input):
    weight_i.append([0.0, 0.0])
weight_o = []
for i in range(n_hidden):
    weight_o.append([0.0])
    
for i in range(n_input):
        for j in range(n_hidden):
            weight_i[i][j] = -1.0 + random.random()*2
        for j in range(n_hidden):
            for k in range(n_output):
                weight_o[j][k] = -1.0 + random.random()*2

inpt = [1.0, 1.0, 1.0]
hidden = [1.0, 1.0]
output = [1.0]
change_i = []
change_o = []
for i in range(n_input):
    change_i.append([0.0, 0.0])
for i in range(n_hidden):
    change_o.append([0.0])


# In[256]:


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


# In[257]:


def sigmoid_derive(x):
    return sigmoid(x)*(1.0-sigmoid(x))


# In[258]:


def update(inputs):
    for i in range(n_input-1):
        inpt[i] = inputs[i]
    for i in range(n_hidden):
        sum = 0.0
        for j in range(n_input):
            sum += inpt[j]*weight_i[j][i]
        hidden[i] = sigmoid(sum)
    for i in range(n_output):
        sum = 0.0
        for j in range(n_hidden):
            sum += hidden[j]*weight_o[j][i]
        output[i] = sigmoid(sum)
    return output[:]


# In[259]:


def back_propagate(target, N, M):
    out = [0.0]
    error = target[0]-output[0]   
    out[0] = sigmoid_derive(output[0])*error
    hid = [0.0, 0.0]
    for i in range(n_hidden):
        error = 0.0
        for j in range(n_output):
            error += out[j]*weight_o[i][j]
        hid[i] = sigmoid_derive(hidden[i])*error
    for i in range(n_hidden):
        for j in range(n_output):
            delta = out[j]*hidden[i]
            weight_o[i][j] += N*delta + M*change_o[i][j]
            change_o[i][j] = delta
    for i in range(n_input):
        for j in range(n_hidden):
            delta = hid[j]*inpt[i]
            weight_i[i][j] += N*delta + M*change_i[i][j]
            change_i[i][j] = delta
    error = 0.0
    for i in range(len(target)):
        error += ((target[i]-output[i])**2)/2
    return error


# In[260]:


accuracy = []
for i in range(100000):
    if (i+1)%10000 == 0:
        predict = []
    error = 0.0
    for d in data:
        inputs = d[0]
        target = d[1]
        u = update(inputs)
        if (i+1)%10000 == 0:
            predict.append(u)
        error += back_propagate(target, 0.01, 0.1)
    if (i+1)%10000==0:
        print('epochs ', i+1, ' loss:', error)
        acc = 0.0
        for j in range(len(data)):
            if (predict[j][0] <= 0.5 and data[j][1][0] == 0) or (predict[j][0] > 0.5 and data[j][1][0] == 1):
                acc += 1
        accuracy.append([acc/len(data)])
print('Accruacy:')
print(accuracy)


# In[261]:


x = []
y = []
for line in groud_truth_0:
    x.append(line[0])
    y.append(line[1])
plt.scatter(x, y)
x = []
y = []
for line in groud_truth_1:
    x.append(line[0])
    y.append(line[1])
plt.scatter(x, y)
plt.title('Ground truth')
plt.show()


# In[251]:


predict_0 = []
predict_1 = []
for i in range(len(data)):
    if predict[i][0] <= 0.5:
        predict_0.append(data[i][0])
    else:
        predict_1.append(data[i][0])
x = []
y = []
for line in predict_0:
    x.append(line[0])
    y.append(line[1])
plt.scatter(x, y)
x = []
y = []
for line in predict_1:
    x.append(line[0])
    y.append(line[1])
plt.scatter(x,y)
plt.title('Predict result')
plt.show()

