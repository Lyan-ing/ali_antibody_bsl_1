import re
import pandas as pd
import matplotlib.pyplot as plt
from ast import Index
from cmath import nan
from operator import index
import numpy as np
from sklearn import linear_model

dataset = pd.read_csv("./data/Neutralization_train_new_raw.csv", encoding='gbk')


# 查找空值的函数
def exit_index_search(a):
    a1 = []
    for i in range(len(a)):
        if np.isnan(a[i]):
            pass
        else:
            a1.append(i)
    return a1


def completion_index_search(a):
    a1 = []
    for i in range(len(a)):
        if np.isnan(a[i]):
            a1.append(i)
        else:
            pass
    return a1


# 对ELISA和SPR的关系进行建模
# 取出SPR的列表
LIVE_frame = dataset[['LIVE_IC50']]
LIVE_list = np.array(LIVE_frame).squeeze().tolist()
LIVE_exit_index = exit_index_search(LIVE_list)
# 找到对应ELISA值
PSE_frame = dataset[['PSE_IC50']]
PSE_list = np.array(PSE_frame).squeeze().tolist()
PSE_exit_index = exit_index_search(PSE_list)
# 找到LIVE和PSE都非空值的索引，便于后续建模
common_exit_index = list(set(LIVE_exit_index) & set(PSE_exit_index))
common_exit_index.sort()
# 取出LIVE的非空子集作为x, 取出PSE的非空子集作为y
x_list = [LIVE_list[i] for i in common_exit_index]
x2 = np.array(x_list).reshape(-1, 1)
y_list = [PSE_list[i] for i in common_exit_index]
y2 = np.array(y_list)
# PSE为nan，LIVE为真，即可进行补充
PSE_nan_index = completion_index_search(PSE_list)
completion_index = list(set(LIVE_exit_index) & set(PSE_nan_index))
completion_index.sort()

# 极限树建模
from sklearn.tree import ExtraTreeRegressor

clf = ExtraTreeRegressor()
model_2 = clf.fit(x2, y2)
y_pred = model_2.predict(x2)
# 待补充数据集的预测输出
x_com = [LIVE_list[i] for i in completion_index]
x_com = np.array(x_com).reshape(-1, 1)
y_com = model_2.predict(x_com)
# 实验值补全
a = 0
for i in completion_index:
    dataset.iloc[i, 3] = y_com[a]
    a += 1
dataset.dropna(axis=0, how='any', inplace=True, subset=['PSE_IC50'])  # 删除所有空值

PSEN_frame = dataset[['PSE_IC50']]
PSEN_list = np.array(PSEN_frame).squeeze().tolist()


# convert labels
def convert_label(_list):
    res = []
    for item in _list:
        lab = 0
        if item < 0.01:
            lab = 5
        elif 0.01 <= item < 0.1:
            lab = 4
        elif 0.1 <= item < 0.25:
            lab = 3
        elif 0.25 <= item < 1:
            lab = 2
        elif item >= 1:
            lab = 1
        res.append(lab)
    return res


dataset.insert(dataset.shape[1], 'Label', convert_label(PSEN_list))
# select train data
sequence = dataset['Sequence'].tolist()
label = dataset['Label'].tolist()
sequence_new = []
label_new = []
for i, elem in enumerate(sequence):
    if elem.find('nan') == -1:
        temp = elem.replace('+', '').replace(' ', '').replace('\n', '').replace('2', '')\
            .replace('5', '').replace('1', '').replace('3', '').replace('4', '').replace('6', '')\
            .replace('7', '').replace('8', '').replace('9', '').replace('0', '')
        sequence_new.append(temp)
        label_new.append(label[i])

df2 = pd.concat([pd.DataFrame({'Sequence': sequence_new}), pd.DataFrame({'Label': label_new})], axis=1)
print(df2)
df2.to_csv('./data/Neutralization_train_data.csv', index=False)
