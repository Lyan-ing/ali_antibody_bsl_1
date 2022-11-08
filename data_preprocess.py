# 1.1读取csv数据集，并展示其head
import re
import pandas as pd
import numpy as np
from ast import Index
from cmath import nan
from operator import index


# get corresponding antigen sequence
SARS = ["Alpha", "Beta", "Delta", "Gamma", "Kappa", "Omicron","SARS-Cov1", "SARS-CoV2_WT"]
def get_antigen(_name):
    if _name not in SARS:
        fp = open('./data/SARS-CoV2_WT.fasta')
    else:
        fp = open('./data/%s.fasta' % _name)
    antigen_seq = ''
    for lne in fp:
        if not lne.startswith('>'):
            antigen_seq += lne.strip('\n')
    fp.close()
    return antigen_seq


# 定义读取性能指标的函数，剥离[单位、作者、时间]等建模无用信息
def label_split(exp_index):
    a = re.findall("\d+\.?\d*", str(exp_index))  # 正则表达式
    if a:
        return a[0]
    else:
        return float('nan')


# 用于剥离一列数据指标的函数
def label_list_split(_label_list):
    _index = []
    for i in _label_list:
        if pd.isna(i):
            _index.append(i)
        else:
            i1 = label_split(i)
            _index.append(float(i1))
    return _index


# 列表拼接和最小数提取
# 定义函数查找带空值的数组中的最小值
def min_search(Mat):
    min_list = []
    for elem in Mat:
        min_index = np.nanmin(elem)
        min_list.append(min_index)
    return min_list


# 剥离SPR，BLI，ELISA，FACS实验中各个列的数值并合并（取最小值）
def make_data_list(_exp_list, _dataset):
    index_list_all = []
    for _exp_name in _exp_list:
        label_df = _dataset[[_exp_name]]
        label_list = np.array(label_df).squeeze().tolist()
        index_list = label_list_split(label_list)
        index_list_all.append(index_list)
    zip_start = index_list_all[0]
    data_list = []
    for list_elem in index_list_all[1:]:
        exp_ZIP = zip(zip_start, list_elem)
        exp_Mat = list(exp_ZIP)
        data_list = min_search(exp_Mat)
        zip_start = data_list
    return data_list


# 根据输入的亲和力train.csv文件，提取抗体的序列信息和各个实验数值
def generate_affinity_rawdata(_filename):
    dataset = pd.read_csv("./data/%s" % _filename, encoding='gbk')

    # 读取抗体对应的名称，亦即“Name”
    antibody_name = dataset[['Name']]
    antibody_list = np.array(antibody_name).squeeze().tolist()

    # 读取抗体对应的序列，分别提取VHH、VL、CDRH3以及CDRL3
    dataset = dataset.replace('ND', '')
    antibody_seq1 = dataset['VH or VHH'].astype(str).tolist()
    antibody_seq2 = dataset['VL'].astype(str).tolist()
    antibody_seq3 = dataset['CDRH3'].astype(str).tolist()
    antibody_seq4 = dataset['CDRL3'].astype(str).tolist()
    antigen = dataset['Binds to'].astype(str).tolist()
    atg_set = list(set(antigen))

    # make a dic: antigen_name : sequence
    atg_dic = {}
    for item in atg_set:
        seq = get_antigen(item)
        atg_dic[item] = seq

    # 将抗体序列合并，作为训练的输入
    length = len(antibody_seq1)

    # 然后将它们合并起来，作为单一维度的输入
    merge_seq = []
    for i in range(length):
        # 判断使用RBD区的序列还是NTD区的序列后再分别处理
        merge_seq.append(atg_dic[antigen[i]][329:583] + str(antibody_seq1[i]) + str(antibody_seq2[i])
                         + str(antibody_seq3[i]) + str(antibody_seq4[i]))

    SPR_exp_list = ['SPR RBD (KD; nm)', 'SPR S1 (KD; nm)', 'SPR S2 (KD; nm)', 'SPR S-ECD (KD; nm)',
                    'SPR S (KD; nm)', 'SPR NTD (KD; nm)', 'SPR N (KD; nm)']
    BLI_exp_list = ['BLI RBD (KD; nm)', 'BLI S1 (KD; nm)', 'BLI S (KD; nm)', 'BLI NTD (KD; nm)',
                    'BLI N (KD; nm)']
    ELISA_exp_list = ['ELISA RBD competitive (IC50; μg/ml)', 'ELISA S1 competitive (IC50; μg/ml)',
                      'ELISA S competitive (IC50; μg/ml)', 'ELISA S competitive (IC80; μg/ml)',
                      'ELISA NTD competitive (IC50; μg/ml)', 'ELISA RBD binding (EC50; μg/ml)',
                      'ELISA S1 binding (EC50; μg/ml)', 'ELISA S binding (EC50; μg/ml)',
                      'ELISA N binding (EC50; μg/ml)']
    FACS_exp_list = ['FACS RBD (IC50; nm/ml)', 'FACS S (IC50; nm/ml)']
    SPR_list = make_data_list(SPR_exp_list, dataset)
    FACS_list = make_data_list(FACS_exp_list, dataset)
    BLI_list = make_data_list(BLI_exp_list, dataset)
    ELISA_list = make_data_list(ELISA_exp_list, dataset)

    # 把之前提取的输入和输出保存成CSV
    dataframe = pd.DataFrame({'Name': antibody_list, 'Sequence': merge_seq, 'SPR': SPR_list,
                              'BLI': BLI_list, 'ELISA': ELISA_list, 'FCAS': FACS_list})
    _name = _filename.split('.')[0]
    dataframe.to_csv(r"./data/%s_raw.csv" % _name, sep=',', index=False)


def generate_neutralization_rawdata(_filename):
    dataset = pd.read_csv("./data/%s" % _filename, encoding='gbk')
    # 读取抗体对应的名称，亦即“Name”
    antibody_name = dataset[['Name']]
    antibody_list = np.array(antibody_name).squeeze().tolist()

    # 读取抗体对应的序列，分别提取VHH、VL、CDRH3以及CDRL3
    dataset = dataset.replace('ND', '')
    antibody_seq1 = dataset['VH or VHH'].astype(str).tolist()
    antibody_seq2 = dataset['VL'].astype(str).tolist()
    antibody_seq3 = dataset['CDRH3'].astype(str).tolist()
    antibody_seq4 = dataset['CDRL3'].astype(str).tolist()
    antigen = dataset['Neutralising Vs'].astype(str).tolist()
    atg_set = list(set(antigen))

    # make a dic: antigen_name : sequence
    atg_dic = {}
    for item in atg_set:
        seq = get_antigen(item)
        atg_dic[item] = seq

    # 将抗体序列合并，作为训练的输入
    length = len(antibody_seq1)
    # 然后将它们合并起来，作为单一维度的输入
    merge_seq = []
    for i in range(length):
        merge_seq.append(atg_dic[antigen[i]][329:583] + str(antibody_seq1[i]) + str(antibody_seq2[i])
                         + str(antibody_seq3[i]) + str(antibody_seq4[i]))
    # label的剥离:live virus IC50和pseudo virus IC50
    # # 1. LIVE_IC50
    # LIVE_IC50_label_df = dataset[['Live Virus Neutralisation IC50 (50% titre; μg/ml)阈值2μg/ml']]
    # LIVE_IC50_label_list = np.array(LIVE_IC50_label_df).squeeze().tolist()
    # LIVE_IC50_index_list = label_list_split(LIVE_IC50_label_list)
    # # 2. PSE_IC50
    # PSE_IC50_label_df = dataset[['Pseudo Virus Neutralisation IC50 (50% titre; μg/ml)']]
    # PSE_IC50_label_list = np.array(PSE_IC50_label_df).squeeze().tolist()
    # PSE_IC50_index_list = label_list_split(PSE_IC50_label_list)
    LIVE_exp_list = ['Live Virus Neutralisation IC50 (50% titre; μg/ml)阈值2μg/ml',
                     'Live Virus Neutralisation IC80 (80% titre; μg/ml)',
                     'Live Virus Neutralisation IC90 (90% titre; μg/ml)',
                     'Live Virus Neutralisation IC100 (100% titre; μg/ml)']
    PSE_exp_list = ['Pseudo Virus Neutralisation IC50 (50% titre; μg/ml)',
                    'Pseudo Virus Neutralisation IC80 (80% titre; μg/ml)',
                    'Pseudo Virus Neutralisation IC90 (90% titre; μg/ml)',
                    'Pseudo Virus Neutralisation IC100 (100% titre; μg/ml)',
                    'Pseudo Virus Neutralisation (fold change)']
    LIVE_IC_list = make_data_list(LIVE_exp_list, dataset)
    PSE_IC_list = make_data_list(PSE_exp_list, dataset)
    dataframe = pd.DataFrame({'Name': antibody_list, 'Sequence': merge_seq,
                              'LIVE_IC50': LIVE_IC_list,
                              'PSE_IC50': PSE_IC_list})
    _name = _filename.split('.')[0]
    dataframe.to_csv(r"./data/%s_raw.csv" % _name, sep=',', index=False)


if __name__ == '__main__':
    generate_affinity_rawdata('Affinity_train_new.csv')
    generate_affinity_rawdata('Affinity_extraTrainData_new.csv')
    generate_neutralization_rawdata('Neutralization_train_new.csv')
