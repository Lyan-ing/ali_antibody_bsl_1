import pandas as pd
import torch
import numpy as np
from tape import ProteinBertModel, TAPETokenizer
from tqdm import tqdm


def get_feature(_list):
    # load model
    # model = ProteinBertModel.from_pretrained('bert-base')
    # torch.save(model, 'pretrain_bert.models')
    model = torch.load('pretrain_bert.models', map_location='cpu')
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model

    feature = []
    token_ids = 0
    for seq in tqdm(_list):
        token_ids = torch.tensor([tokenizer.encode(seq)])
        output = model(token_ids)
        pooled_output = output[1]
        feature.append(pooled_output[0].tolist())

    _df = pd.DataFrame(np.array(feature))
    return _df


if __name__ == '__main__':
    # 提取特征时全部提取
    df = pd.read_csv('./data/Affinity_train_data.csv', sep=',')
    sequence = df['Sequence'].astype(str).tolist()
    df = get_feature(sequence)
    df.to_csv('./data/Affinity_feature.csv', index=False)
    df2 = pd.read_csv('./data/Neutralization_train_data.csv', sep=',')
    sequence = df2['Sequence'].astype(str).tolist()
    df2 = get_feature(sequence)
    df2.to_csv('./data/Neutralization_feature.csv', index=False)
