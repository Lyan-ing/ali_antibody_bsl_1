import pandas as pd
from feature_extract import get_feature
import joblib
import numpy as np
from data_preprocess import get_antigen


def get_input(_file, is_affinity):
    df = pd.read_csv(_file, sep=',', encoding='gbk')
    antibody_seq1 = df['VH or VHH'].astype(str).tolist()
    antibody_seq2 = df['VL'].astype(str).tolist()
    antibody_seq3 = df['CDRH3'].astype(str).tolist()
    antibody_seq4 = df['CDRL3'].astype(str).tolist()
    if is_affinity:
        antigen = df['Binds to'].astype(str).tolist()
    else:
        antigen = df['Neutralising Vs'].astype(str).tolist()
    atg_set = list(set(antigen))
    atg_sets = []
    for atg_set_unique in atg_set:
        [atg_sets.append(atg_) for atg_ in atg_set_unique.split(';')]
    # make a dic: antigen_name : sequence
    atg_set = list(set(atg_sets))
    atg_dic = {}
    for item in atg_set:
        seq = get_antigen(item)
        atg_dic[item] = seq
    # create model input
    merge_seq = []
    length = len(antibody_seq1)
    SARS = ["Alpha", "Beta", "Delta", "Gamma", "Kappa", "Omicron", "SARS-Cov1", "SARS-CoV2_WT"]
    for i in range(length):
        antigen_i = antigen[i].split(";")[0]
        if antigen_i not in SARS:
            antigen_i = "SARS-CoV2_WT"
        if "RBD":
            seq = atg_dic[antigen_i][329:583] + antibody_seq1[i] + antibody_seq2[i] + antibody_seq3[i] + antibody_seq4[i]
        else:
            seq = atg_dic[antigen_i][14:303] + antibody_seq1[i] + antibody_seq2[i] + antibody_seq3[i] + antibody_seq4[
                i]
        seq = seq.replace(' ', '').replace('nan', '').replace('l', '')
        print(len(seq))
        merge_seq.append(seq)
    return merge_seq


if __name__ == '__main__':
    seq_affinity = get_input('./data/Affinity_test.csv', True)
    # seq_neutralization = get_input('./tcdata/Neutralization_test.csv', False)
    # feature extract
    affinity_feature = get_feature(seq_affinity)
    model_affinity = joblib.load('affinity.model')
    affinity_predict_unlabeled = model_affinity.predict(affinity_feature)
    pred_affinity = np.round(affinity_predict_unlabeled)

    # neutralization_feature = get_feature(seq_neutralization)
    model_neutralization = joblib.load('neutralization.model')
    # affinity_predict_unlabeled = model_neutralization.predict(neutralization_feature)
    pred_neutralization = np.round(affinity_predict_unlabeled)

    # save results
    df_result = pd.concat([pd.DataFrame({'label_a': pred_affinity}), pd.DataFrame({'label_n': pred_neutralization})], axis=1)
    df_result.to_csv('result.csv', index=False)
