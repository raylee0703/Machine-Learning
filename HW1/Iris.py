import math
import pandas as pd
import numpy as np
df = pd.read_csv('iris.data')
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

#shuffle and seperate data
df = df.sample(frac=1).reset_index(drop=True)
train = df.sample(frac=0.7, random_state=200)
test = df.drop(train.index)
dataset_train = train.values.tolist()
dataset_test = test.values.tolist()
dataset = df.values.tolist()

#count mean and SD and number of each class
Setosa=0
Versicolour=0
Virginica=0
sepal_l_mean = [0.0, 0.0, 0.0]
sepal_w_mean = [0.0, 0.0, 0.0]
petal_l_mean = [0.0, 0.0, 0.0]
petal_w_mean = [0.0, 0.0, 0.0]
sepal_l_sd = [0.0, 0.0, 0.0]
sepal_w_sd = [0.0, 0.0, 0.0] 
petal_l_sd = [0.0, 0.0, 0.0]
petal_w_sd = [0.0, 0.0, 0.0]
for example in dataset_train:
    if example[4] == 'Iris-setosa':
        Setosa += 1
        sepal_l_mean[0] += example[0]
        sepal_w_mean[0] += example[1]
        petal_l_mean[0] += example[2]
        petal_w_mean[0] += example[3]
    elif example[4] == 'Iris-versicolor':
        Versicolour += 1
        sepal_l_mean[1] += example[0]
        sepal_w_mean[1] += example[1]
        petal_l_mean[1] += example[2]
        petal_w_mean[1] += example[3]
    elif example[4] == 'Iris-virginica':
        Virginica += 1
        sepal_l_mean[2] += example[0]
        sepal_w_mean[2] += example[1]
        petal_l_mean[2] += example[2]
        petal_w_mean[2] += example[3]
sepal_l_mean[0] /= Setosa
sepal_w_mean[0] /= Setosa
petal_l_mean[0] /= Setosa
petal_w_mean[0] /= Setosa
sepal_l_mean[1] /= Versicolour
sepal_w_mean[1] /= Versicolour
petal_l_mean[1] /= Versicolour
petal_w_mean[1] /= Versicolour
sepal_l_mean[2] /= Virginica
sepal_w_mean[2] /= Virginica
petal_l_mean[2] /= Virginica
petal_w_mean[2] /= Virginica
for example in dataset_train:
    if example[4] == 'Iris-setosa':
        sepal_l_sd[0] += (example[0]-sepal_l_mean[0])**2
        sepal_w_sd[0] += (example[1]-sepal_w_mean[0])**2
        petal_l_sd[0] += (example[2]-petal_l_mean[0])**2
        petal_w_sd[0] += (example[3]-petal_w_mean[0])**2
    elif example[4] == 'Iris-versicolor':
        sepal_l_sd[1] += (example[0]-sepal_l_mean[1])**2
        sepal_w_sd[1] += (example[1]-sepal_w_mean[1])**2
        petal_l_sd[1] += (example[2]-petal_l_mean[1])**2
        petal_w_sd[1] += (example[3]-petal_w_mean[1])**2
    elif example[4] == 'Iris-virginica':
        sepal_l_sd[2] += (example[0]-sepal_l_mean[2])**2
        sepal_w_sd[2] += (example[1]-sepal_w_mean[2])**2
        petal_l_sd[2] += (example[2]-petal_l_mean[2])**2
        petal_w_sd[2] += (example[3]-petal_w_mean[2])**2
sepal_l_sd[0] = math.sqrt(sepal_l_sd[0]/(Setosa-1))
sepal_w_sd[0] = math.sqrt(sepal_w_sd[0]/(Setosa-1))
petal_l_sd[0] = math.sqrt(petal_l_sd[0]/(Setosa-1))
petal_w_sd[0] = math.sqrt(petal_w_sd[0]/(Setosa-1))
sepal_l_sd[1] = math.sqrt(sepal_l_sd[1]/(Versicolour-1))
sepal_w_sd[1] = math.sqrt(sepal_w_sd[1]/(Versicolour-1))
petal_l_sd[1] = math.sqrt(petal_l_sd[1]/(Versicolour-1))
petal_w_sd[1] = math.sqrt(petal_w_sd[1]/(Versicolour-1))
sepal_l_sd[2] = math.sqrt(sepal_l_sd[2]/(Virginica-1))
sepal_w_sd[2] = math.sqrt(sepal_w_sd[2]/(Virginica-1))
petal_l_sd[2] = math.sqrt(petal_l_sd[2]/(Virginica-1))
petal_w_sd[2] = math.sqrt(petal_w_sd[2]/(Virginica-1))


def naive_bayes(example):
    P_setosa = Setosa/len(dataset_train)
    P_versicolour = Versicolour/len(dataset_train)
    P_virginica = Virginica/len(dataset_train)
    setosa_prob = math.log(P_setosa)
    versicolour_prob = math.log(P_versicolour)
    virginica_prob = math.log(P_virginica)

    setosa_prob += math.log((1/(sepal_l_sd[0]*math.sqrt(2*math.pi)))*math.exp(-((example[0]-sepal_l_mean[0])**2)/(2*(sepal_l_sd[0]**2))))
    setosa_prob += math.log((1/(sepal_w_sd[0]*math.sqrt(2*math.pi)))*math.exp(-((example[1]-sepal_w_mean[0])**2)/(2*(sepal_w_sd[0]**2))))
    setosa_prob += math.log((1/(petal_l_sd[0]*math.sqrt(2*math.pi)))*math.exp(-((example[2]-petal_l_mean[0])**2)/(2*(petal_l_sd[0]**2))))
    setosa_prob += math.log((1/(petal_w_sd[0]*math.sqrt(2*math.pi)))*math.exp(-((example[3]-petal_w_mean[0])**2)/(2*(petal_w_sd[0]**2))))
    versicolour_prob += math.log((1/(sepal_l_sd[1]*math.sqrt(2*math.pi)))*math.exp(-((example[0]-sepal_l_mean[1])**2)/(2*(sepal_l_sd[1]**2))))
    versicolour_prob += math.log((1/(sepal_w_sd[1]*math.sqrt(2*math.pi)))*math.exp(-((example[1]-sepal_w_mean[1])**2)/(2*(sepal_w_sd[1]**2))))
    versicolour_prob += math.log((1/(petal_l_sd[1]*math.sqrt(2*math.pi)))*math.exp(-((example[2]-petal_l_mean[1])**2)/(2*(petal_l_sd[1]**2))))
    versicolour_prob += math.log((1/(petal_w_sd[1]*math.sqrt(2*math.pi)))*math.exp(-((example[3]-petal_w_mean[1])**2)/(2*(petal_w_sd[1]**2))))
    virginica_prob += math.log((1/(sepal_l_sd[2]*math.sqrt(2*math.pi)))*math.exp(-((example[0]-sepal_l_mean[2])**2)/(2*(sepal_l_sd[2]**2))))
    virginica_prob += math.log((1/(sepal_w_sd[2]*math.sqrt(2*math.pi)))*math.exp(-((example[1]-sepal_w_mean[2])**2)/(2*(sepal_w_sd[2]**2))))
    virginica_prob += math.log((1/(petal_l_sd[2]*math.sqrt(2*math.pi)))*math.exp(-((example[2]-petal_l_mean[2])**2)/(2*(petal_l_sd[2]**2))))
    virginica_prob += math.log((1/(petal_w_sd[2]*math.sqrt(2*math.pi)))*math.exp(-((example[3]-petal_w_mean[2])**2)/(2*(petal_w_sd[2]**2))))


    if setosa_prob >= versicolour_prob and setosa_prob >= virginica_prob:
        return 'Iris-setosa'
    elif versicolour_prob >= setosa_prob and versicolour_prob >= virginica_prob:
        return 'Iris-versicolour'
    else:
        return 'Iris-virginica'

#main
correct = 0
setosa_setosa=0
setosa_vir=0
setosa_ver=0
vir_vir=0
vir_setosa=0
vir_ves=0
ves_ves=0
ves_vir=0
ves_setosa=0
for ex in dataset_test:
    actual = ex[4]
    cal = naive_bayes(ex)
    if actual == 'Iris-setosa' and cal == 'Iris-setosa':
        correct += 1
        setosa_setosa += 1
    elif actual == 'Iris-virginica' and cal == 'Iris-virginica':
        correct += 1
        vir_vir += 1
    elif actual == 'Iris-versicolour' and cal == 'Iris-versicolour':
        correct += 1
        ves_ves += 1
    elif actual == 'Iris-setosa' and cal == 'Iris-virginica':
        setosa_vir += 1
    elif actual == 'Iris-setosa' and cal == 'Iris-versicolour':
        setosa_ver += 1
    elif actual == 'Iris-virginica' and cal == 'Iris-setosa':
        vir_setosa += 1
    elif actual == 'Iris-virginica' and cal == 'Iris-versicolour':
        vir_ves += 1
    elif actual == 'Iris-versicolour' and cal == 'Iris-setosa':
        ves_setosa += 1
    else:
        ves_vir += 1
        
print ("                  Predicted Setosa    Predicted Virginica    Predicted Versicolour")
print ("Actual Setosa           ", setosa_setosa, "                 ", setosa_vir, "                   ", setosa_ver)
print ("Actual Virginica        ", vir_setosa, "                  ", vir_vir, "                  ", vir_ves)
print ("Actual Versicolour      ", ves_setosa, "                  ", ves_vir, "                  ", ves_ves)
print ('Accuracy: %f' % (float(correct)/float(len(dataset_test))))
print ('Precision (Setosa = True): %f' %(float(setosa_setosa)/(float(setosa_setosa)+float(setosa_vir)+float(setosa_ver)))) 
print ('Sensitivity (Setosa = True): %f' %(float(setosa_setosa)/(float(setosa_setosa)+float(vir_setosa)+float(ves_setosa)))) 
print ('Precision (Virginica = True): %f' %(float(vir_vir)/(float(vir_vir)+float(vir_setosa)+float(vir_ves))))
print ('Sensitivity (Virginica = True): %f' %(float(vir_vir)/(float(vir_vir)+float(setosa_vir)+float(ves_vir))))
print ('Precision (Versicolour = True): %f' %(float(ves_ves)/(float(ves_ves)+float(ves_setosa)+float(ves_vir))))
print ('Sensitivity (Versicolour = True): %f' %(float(ves_ves)/(float(ves_ves)+float(setosa_ver)+float(vir_ves))))