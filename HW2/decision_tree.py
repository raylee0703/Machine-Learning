import numpy as np
import pandas as pd
import math
import csv


class Node(object):
    def __init__(self):
        self.attr = None
        self.decision = None
        self.childs = None


df = pd.read_csv("x_train.csv")
cat = pd.read_csv("y_train.csv")

drop_list = ['workclass', 'occupation', 'native-country']
df = df.drop(drop_list, axis=1)
#drop education level since it can be replaced by education num
df = df.drop('education', axis=1)
ms = {' Married-civ-spouse':0, ' Divorced':1, ' Never-married':2, ' Separated':3, ' Widowed':4, ' Married-spouse-absent':5, ' Married-AF-spouse':6}
df['marital-status'] = [ms[item] for item in df['marital-status']]
relation = {' Wife':0, ' Own-child':1, ' Husband':2, ' Not-in-family':3, ' Other-relative':4, ' Unmarried':5}
df['relationship'] = [relation[item] for item in df['relationship']]
race = {' White':0, ' Asian-Pac-Islander':1, ' Amer-Indian-Eskimo':2, ' Other':3, ' Black':4}
df['race'] = [race[item] for item in df['race']]
sex = {' Female':0, ' Male':1 }
df['sex'] = [sex[item] for item in df['sex']]
df = pd.merge(cat, df, how='left', on='Id')

attribute = df.columns.tolist()
#shuffle data
df = df.sample(frac=1).reset_index(drop=True)
train = df.sample(frac=0.7, random_state=200)
train = train.reset_index()
validation = df.drop(train.index)



def entropy(df, attribute, attr):
    
    count_list = pd.value_counts(df[attr].values, sort=False).tolist()
    data_entropy = 0.0
    for entry in count_list:
        data_entropy += (-1*entry/len(df))*math.log(entry/len(df), 2)

    return data_entropy


def remainder(df, subsets, attribute, attr):
    re = 0.0
    for sub in subsets:
        if len(sub) > 1:
            re += float(len(sub)/len(df))*entropy(sub, attribute, attr)
    return re


def info_gain(df, attribute, attr, thres):
    st_thres = pd.DataFrame()
    lt_thres = pd.DataFrame()
    for i in range(0, len(df)-1):
        example = df.loc[i]
        if example[attr] < thres:
            st_thres = st_thres.append(example)
        else:
            lt_thres = lt_thres.append(example)
    ig = entropy(df, attribute, attr) - remainder(df, [st_thres, lt_thres], attribute, attr)
    return ig


def find_threshold(df, attribute, attr):
    values = df[attr].tolist()
    values = [ float(x) for x in values]
    values.sort()
    values = list(dict.fromkeys(values))
    
    max_ig = float("-inf")
    threshold = 0.0
    for i in range(0, len(values) - 1):
        thres = (values[i] + values[i+1])/2
        ig_temp = info_gain(df, attribute, attr, thres)
        if ig_temp > max_ig:
            max_ig = ig_temp
            threshold = thres
    return threshold


def CalMaxGain(df, attribute):
    maxGain = 0
    retattr = ''
    k=0
    attr_temp = attribute
    if 'Id' in attr_temp:
        attr_temp.remove('Id')
        attr_temp.remove('Category')
    for attr in attr_temp:
        en = entropy(df, attribute, attr)
        count_dict = {}
        df = df.reset_index(drop = True)
        k+=1
        for i in range(0, len(df)):
            temp = df.loc[i, attr]
            if temp not in count_dict:
                count_dict[temp] = 1
            else:
                count_dict[temp] += 1

        gain = en

        for key in count_dict:
            yes = 0
            no = 0
            for j in range(0, len(df)):
                if df.loc[j, attr] == key:
                    if df.loc[j, 'Category'] == 1:
                        yes = yes + 1
                    else:
                        no = no +1
            y_ratio = yes/(yes+no)
            n_ratio = no/(yes+no)
            if y_ratio !=0 and n_ratio != 0:
                gain += (count_dict[key]*(y_ratio*math.log(y_ratio, 2)+n_ratio*math.log(n_ratio, 2)))/len(df)

            if gain >= maxGain:
                maxGain = gain
                retattr = attr

    return maxGain, retattr 


def tree_build(df, attribute):
    maxGain, attr_sel = CalMaxGain(df, attribute)
    root = Node()
    root.childs = []
    root.attr = attr_sel
    df = df.reset_index(drop = True)

    if attr_sel == '':
        yes=0
        no=0
        for i in range(0, len(df)):
            if df.loc[i, 'Category'] == 0:
                no += 1
            else:
                yes += 1
        if no > yes:
            root.attr = 'No'
        else:
            root.attr = 'Yes'
        return root

    mydict = {}

    for i in range(0, len(df)):
        key = df.loc[i, attr_sel]
        if key not in mydict:
            mydict[key] = 1
        else:
            mydict[key] += 1

    new_attr = attribute
    new_attr.remove(attr_sel)

    for key in mydict:
        newdf = pd.DataFrame()
        for i in range(0, len(df)):
            if df.loc[i, attr_sel] == key:
                newdf = newdf.append(df.loc[i])
        temp = tree_build(newdf, new_attr)
        temp.decision = key
        root.childs.append(temp)
    return root


age_thres = find_threshold(train, attribute, 'age')
train.loc[train['age'] >= age_thres, 'age'] = 1
train.loc[train['age'] < age_thres, 'age'] = 0

fnlwgt_thres = find_threshold(train, attribute, 'fnlwgt')
train.loc[train['fnlwgt'] >= fnlwgt_thres, 'fnlwgt'] = 1
train.loc[train['fnlwgt'] < fnlwgt_thres, 'fnlwgt'] = 0

edunum_thres = find_threshold(train, attribute, 'education-num')
train.loc[train['education-num'] >= edunum_thres, 'education-num'] = 1
train.loc[train['education-num'] < fnlwgt_thres, 'education-num'] = 0

capgain_thres = find_threshold(train, attribute, 'capital-gain')
train.loc[train['capital-gain'] >= capgain_thres, 'capital-gain'] = 1
train.loc[train['capital-gain'] < capgain_thres, 'capital-gain'] = 0

caploss_thres = find_threshold(train, attribute, 'capital-loss')
train.loc[train['capital-loss'] >= caploss_thres, 'capital-loss'] = 1
train.loc[train['capital-loss'] < caploss_thres, 'capital-loss'] = 0

hours_thres = find_threshold(train, attribute, 'hours-per-week')
train.loc[train['hours-per-week'] >= hours_thres, 'hours-per-week'] = 1
train.loc[train['hours-per-week'] < hours_thres, 'hours-per-week'] = 0


def predict(node, data):
    if node.attr == 'Yes':
        return 'Yes'
    elif node.attr == 'No':
        return 'No'
    else:
        for temp in node.childs:
                if temp.decision == data[node.attr]:
                    return predict(temp,data)

def validate(root, df):
    correct = 0
    TP=0
    FN=0
    FP=0
    TN=0
    for i in range(0, len(df)):
        pred = predict(root, df.loc[i])
        if pred == 'Yes' and df.loc[i, 'Category'] == 1:
            correct += 1
            TP+=1
        elif pred == 'No' and df.loc[i, 'Category'] == 0:
            correct += 1
            TN+=1
        elif pred == 'Yes' and df.loc[i, 'Category'] == 0:
            FP+=1
        else:
            FN+=1
    confusion_mat = {'Predicted Positive': [TP, FP],
                     'Predicted Negative': [FN, TN]}
    matrix = pd.DataFrame(confusion_mat)
    matrix.index = ['Target Postive', 'Target Negative']
    print(matrix)
    print ('Accuracy: %f' % (float(correct)/float(len(df))))
    if TP == 0:
        print ('Sensitivity: 0.000000')
        print ('Precision: 0.000000')
    else:
        print ('Sensitivity: %f' % (float(TP)/(float(TP)+float(FN))))
        print ('Precision: %f' % (float(TP)/(float(TP)+float(FP))))

root = tree_build(train, attribute)


validation.loc[validation['age'] >= age_thres, 'age'] = 1
validation.loc[validation['age'] < age_thres, 'age'] = 0
validation.loc[validation['fnlwgt'] >= fnlwgt_thres, 'fnlwgt'] = 1
validation.loc[validation['fnlwgt'] < fnlwgt_thres, 'fnlwgt'] = 0
validation.loc[validation['education-num'] >= edunum_thres, 'education-num'] = 1
validation.loc[validation['education-num'] < fnlwgt_thres, 'education-num'] = 0
validation.loc[validation['capital-gain'] >= capgain_thres, 'capital-gain'] = 1
validation.loc[validation['capital-gain'] < capgain_thres, 'capital-gain'] = 0
validation.loc[validation['capital-loss'] >= caploss_thres, 'capital-loss'] = 1
validation.loc[validation['capital-loss'] < caploss_thres, 'capital-loss'] = 0
validation.loc[validation['hours-per-week'] >= hours_thres, 'hours-per-week'] = 1
validation.loc[validation['hours-per-week'] < hours_thres, 'hours-per-week'] = 0

validation = validation.reset_index(drop=True)

validate(root, validation)

'''
#competition
test = pd.read_csv('X_test.csv')
test = test.reset_index(drop=True)
test.loc[test['age'] >= age_thres, 'age'] = 1
test.loc[test['age'] < age_thres, 'age'] = 0
test.loc[test['fnlwgt'] >= fnlwgt_thres, 'fnlwgt'] = 1
test.loc[test['fnlwgt'] < fnlwgt_thres, 'fnlwgt'] = 0
test.loc[test['education-num'] >= edunum_thres, 'education-num'] = 1
test.loc[test['education-num'] < fnlwgt_thres, 'education-num'] = 0
test.loc[test['capital-gain'] >= capgain_thres, 'capital-gain'] = 1
test.loc[test['capital-gain'] < capgain_thres, 'capital-gain'] = 0
test.loc[test['capital-loss'] >= caploss_thres, 'capital-loss'] = 1
test.loc[test['capital-loss'] < caploss_thres, 'capital-loss'] = 0
test.loc[test['hours-per-week'] >= hours_thres, 'hours-per-week'] = 1
test.loc[test['hours-per-week'] < hours_thres, 'hours-per-week'] = 0
test['marital-status'] = [ms[item] for item in test['marital-status']]
test['relationship'] = [relation[item] for item in test['relationship']]
test['race'] = [race[item] for item in test['race']]
test['sex'] = [sex[item] for item in test['sex']]
with open('result.csv', 'w', newline='') as csvFile:

    for i in range(0, len(test)):
        pred = predict(root, test.loc[i])
        result = 0
        if pred == 'Yes':
            result = 1
        else:
            result = 0
        ans=[]
        ans.append(test.loc[i, 'Id'])
        ans.append(result)
        writer = csv.writer(csvFile)
        writer.writerow(ans)
csvFile.close()
'''