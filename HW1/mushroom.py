import math
import pandas as pd
import numpy as np
df = pd.read_csv('agaricus-lepiota.data')
df.columns = ['edible', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

#process missing feature
df = df.replace('?', np.nan)
thresh = len(df)
df.dropna(thresh = thresh, axis = 1, inplace = True)
df = df.sample(frac=1).reset_index(drop=True)
#shuffle and seperate data
train = df.sample(frac=0.7, random_state=200)
test = df.drop(train.index)
dataset_train = train.values.tolist()
dataset_test = test.values.tolist()
dataset = df.values.tolist()


attribute = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
attribute_dic = {'cap-shape': ['b', 'c', 'x', 'f', 'k', 's'], 'cap-surface': ['f', 'g', 'y', 's'], 'cap-color': ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'], 'bruises': ['t', 'f'], 'odor': ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'], 'gill-attachment': ['a', 'd', 'f', 'n'], 'gill-spacing': ['c', 'w', 'd'], 'gill-size': ['b', 'n'], 'gill-color': ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'], 'stalk-shape': ['e', 't'], 'stalk-surface-above-ring': ['f', 'y', 'k', 's'], 'stalk-surface-below-ring': ['f', 'y', 'k', 's'], 'stalk-color-above-ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'], 'stalk-color-below-ring': ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'], 'veil-type': ['p', 'u'], 'veil-color': ['n', 'o', 'w', 'y'], 'ring-number': ['n', 'o', 't'], 'ring-type': ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'], 'spore-print-color': ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'], 'population': ['a', 'c', 'n', 's', 'v', 'y'], 'habitat': ['g', 'l', 'm', 'p', 'u', 'w', 'd']}
attribute_pos=[]
attribute_neg=[]
attr_count = 0
val_count = 0
for i in range(len(attribute)):
    attribute_pos.append([])
    attribute_neg.append([])

for i in attribute_pos:
    for j in range(12):
        i.append(0)
        
for i in attribute_neg:
    for j in range(12):
        i.append(0)

for attr in attribute:
    val_count = 0
    for value in attribute_dic[attr]:
        for example in dataset_train:
            if (value == example[attr_count+1]) and example[0] == 'e':
                attribute_pos[attr_count][val_count] += 1
            elif (value == example[attr_count+1]) and example[0] == 'p':
                attribute_neg[attr_count][val_count] += 1
        val_count += 1
    attr_count += 1
attr_count = 0

feature_count=[]
for i in attribute:
    feature_count.append(len(attribute_dic[i]))

def naive_bayes(example, neg, pos):
    count = 0

    PY = pos/len(dataset_train)    
    PY_bar = neg/len(dataset_train)
    pos_prob = math.log(PY)
    neg_prob = math.log(PY_bar)
    
    example.pop(0)
    for attr in example:
        p = attribute_pos[count][attribute_dic[attribute[count]].index(attr)]/pos
        n = attribute_neg[count][attribute_dic[attribute[count]].index(attr)]/neg
        if p != 0:
            pos_prob += math.log(p)
        if n != 0:
            neg_prob += math.log(n)
        count += 1

    if neg_prob > pos_prob:
        return 'p'
    else:
        return 'e'

def naive_bayes_smoothing(example, neg, pos):
    count = 0
    k=1

    PY = pos/len(dataset_train)    
    PY_bar = neg/len(dataset_train)
    pos_prob = math.log(PY)
    neg_prob = math.log(PY_bar)
    
    for attr in example:
        p = (attribute_pos[count][attribute_dic[attribute[count]].index(attr)]+k)/(pos+k*feature_count[count])
        n = (attribute_neg[count][attribute_dic[attribute[count]].index(attr)]+k)/(neg+k*feature_count[count])
        if p != 0:
            pos_prob += math.log(p)
        if n != 0:
            neg_prob += math.log(n)
        count += 1

    if neg_prob > pos_prob:
        return 'p'
    else:
        return 'e'

num_pos = 0
num_neg = 0
pos_train = []
neg_train = []

for i in dataset_train: 
    if i[0] == 'e':
        num_pos += 1
        pos_train.append(i[1])
    else:
        num_neg += 1
        neg_train.append(i[1])
correct = 0
edible_edible=0
edible_poisonous=0
poisonous_edible=0
poisonous_poisonous=0
correct_smoothing = 0
edible_edible_s=0
edible_poisonous_s=0
poisonous_edible_s=0
poisonous_poisonous_s=0
for ex in dataset_test:
    actual = ex[0]
    cal = naive_bayes(ex, num_neg, num_pos)
    cal_smoothing = naive_bayes_smoothing(ex, num_neg, num_pos)
    if actual == 'e' and cal == 'e':
    	correct += 1
    	edible_edible += 1
    elif actual == 'p' and cal == 'p':
    	correct += 1
    	poisonous_poisonous += 1
    elif actual == 'e' and cal == 'p':
    	edible_poisonous += 1
    else:
    	poisonous_edible += 1
    
    if actual == 'e' and cal_smoothing == 'e':
    	correct_smoothing +=1
    	edible_edible_s += 1
    elif actual == 'p' and cal_smoothing == 'p':
    	correct_smoothing += 1
    	poisonous_poisonous_s += 1
    elif actual == 'e' and cal_smoothing == 'p':
    	edible_poisonous_s += 1
    else:
    	poisonous_edible_s += 1
    '''
    if actual == cal:
        correct += 1
    if actual == cal_smoothing:
        correct_smoothing += 1
	'''
print ('Result without Laplace smoothing:')
print ("                Predicted edible 	Predicted poisonous")
print ("Actual edible         ", edible_edible, "                   ", edible_poisonous)
print ("Actual poisonous      ", poisonous_edible, "                   ", poisonous_poisonous)
print ('Accuracy: %f' % (float(correct)/float(len(dataset_test))))
print ('Sensitivity: %f' % (float(edible_edible)/(float(edible_edible)+float(edible_poisonous))))
print ('Precision: %f' % (float(edible_edible)/(float(edible_edible)+float(poisonous_edible))))

print ('\nResult with Laplace smoothing(k=1): ')
print ("                Predicted edible 	Predicted poisonous")
print ("Actual edible         ", edible_edible_s, "                   ", edible_poisonous_s)
print ("Actual poisonous      ", poisonous_edible_s, "                   ", poisonous_poisonous_s)
print ('Accuracy: %f' % (float(correct_smoothing)/float(len(dataset_test))))
print ('Sensitivity: %f' % (float(edible_edible_s)/(float(edible_edible_s)+float(edible_poisonous_s))))
print ('Precision: %f' % (float(edible_edible_s)/(float(edible_edible_s)+float(poisonous_edible_s))))

