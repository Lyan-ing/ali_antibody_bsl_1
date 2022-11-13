import re
import pandas as pd
import matplotlib.pyplot as plt
from ast import Index
from cmath import nan
from operator import index
import numpy as np
from sklearn import linear_model

dataset = pd.read_csv("./data/Affinity_extraTrainData_new_raw.csv", encoding='gbk')


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
df_full_ext = dataset.loc[:, ["Name", "Sequence", "SPR"]]
df_full_ext.dropna(axis=0, how='any', inplace=True)  # 删除所有空值

# 对ELISA和SPR的关系进行建模
# 取出SPR的列表
SPR_frame = dataset[['SPR']]
SPR_list = np.array(SPR_frame).squeeze().tolist()


# 找到对应ELISA值
ELISA_frame = dataset[['ELISA']]
ELISA_list = np.array(ELISA_frame).squeeze().tolist()
ELISA_exit_index = exit_index_search(ELISA_list)
# 拿到ELISA对应的浮点数值
ELISA_exit_score = [ELISA_list[i] for i in ELISA_exit_index]
# 拿到SPR对应的浮点数值
SPR_exit_score = [SPR_list[i] for i in ELISA_exit_index]
# 找到从ELISA到SPR的转化关系，从而用逆运算得到SPR转ELISA的关系，最终补全ELISA
x2 = np.array(ELISA_exit_score).reshape(-1, 1)
y2 = np.array(SPR_exit_score)

# 对SPR的实验值进行补全
# 极限树建模
from sklearn.tree import ExtraTreeRegressor

clf = ExtraTreeRegressor()
model_2 = clf.fit(x2, y2)
y_pred = model_2.predict(x2)
print("训练集分数：", model_2.score(x2,y2))

# ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××
# 找到对应BLI值
BLI_frame = dataset[['BLI']]
BLI_list = np.array(BLI_frame).squeeze().tolist()
BLI_exit_index = exit_index_search(BLI_list)
# 拿到BLI对应的浮点数值
BLI_exit_score = [BLI_list[i] for i in BLI_exit_index]
# 拿到SPR对应的浮点数值
SPR_exit_score_BLI = [SPR_list[i] for i in BLI_exit_index]
# 找到从BLI到SPR的转化关系，从而用逆运算得到SPR转BLI的关系，最终补全BLI
x3 = np.array(BLI_exit_score).reshape(-1, 1)
y3 = np.array(SPR_exit_score_BLI)

# 对SPR的实验值进行补全
# 极限树建模

clf3 = ExtraTreeRegressor()
model_3 = clf3.fit(x3, y3)
y_pred3 = model_3.predict(x3)
print("训练集分数：",model_3.score(x3,y3))
# ××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××

# 在train_full_raw.csv文件夹，首先要保证ELISA的数据都是有值的
dataset_train = pd.read_csv("./data/Affinity_train_new_raw.csv", encoding='gbk')
df_full = dataset_train.loc[:, ["Name", "Sequence", "ELISA"]]
df_full.dropna(axis=0, how='any', inplace=True)  # 删除所有空值

# 根据ELISA的值预测SPR
ELISA_full_frame = df_full[['ELISA']]
ELISA_full_list = np.array(ELISA_full_frame).squeeze().tolist()
z2 = np.array(ELISA_full_list).reshape(-1, 1)
SPR_completion_score = model_2.predict(z2)


# 根据BLI的值预测SPR
df_full_BLI = dataset_train.loc[:, ["Name", "Sequence", "BLI"]]
df_full_BLI.dropna(axis=0, how='any', inplace=True)  # 删除所有空值

BLI_full_frame = df_full_BLI[['BLI']]
BLI_full_list = np.array(BLI_full_frame).squeeze().tolist()
z3 = np.array(BLI_full_list).reshape(-1, 1)
SPR_completion_score_BLI = model_3.predict(z3)



# 根据其他方法建立多种实验结果对应SPR的映射关系


# convert labels
def convert_label(_list):
    res = []
    for item in _list:
        lab = 0
        if item < 0.1:
            lab = 5
        elif 0.1 <= item < 1:
            lab = 4
        elif 1 <= item < 10:
            lab = 3
        elif 10 <= item < 100:
            lab = 2
        elif item >= 100:
            lab = 1
        res.append(lab)
    return res


df_full_ext.insert(df_full_ext.shape[1], 'Label', convert_label(SPR_list))
# 融合两个实验BLI,ELISA测出来的结果：
df_full_BLI.insert(df_full_BLI.shape[1], 'Label', convert_label(SPR_completion_score_BLI))
df_full.insert(df_full.shape[1], 'Label', convert_label(SPR_completion_score))
import pandas as dp
df_merge = dp.concat([df_full_BLI, df_full])
df_merge_new = df_merge.drop_duplicates('Sequence')

df_all_with_ext = dp.concat([df_full_ext, df_merge_new])
df_all_with_ext_nolap = df_all_with_ext.drop_duplicates('Sequence')

# select train data
sequence = df_all_with_ext_nolap['Sequence'].tolist()
label = df_all_with_ext_nolap['Label'].tolist()
sequence_new = []
label_new = []
for i, elem in enumerate(sequence):
    if elem.find('nan') == -1:  # 四种特性有缺失
        temp = elem.replace('+', '').replace(' ', '').replace('\n', '').replace('2', '') \
            .replace('5', '').replace('1', '').replace('3', '').replace('4', '').replace('6', '') \
            .replace('7', '').replace('8', '').replace('9', '').replace('0', '')
        sequence_new.append(temp)
        label_new.append(label[i])

    # else:
    #     print(elem)

df2 = pd.concat([pd.DataFrame({'Sequence': sequence_new}), pd.DataFrame({'Label': label_new})], axis=1)
print(df2)
df2.to_csv('./data/Affinity_train_data_BLI_withEXT.csv', index=False)
