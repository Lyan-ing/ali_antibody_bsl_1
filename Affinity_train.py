import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier
# import xgboost as xgb
from scipy.stats import pearsonr
import joblib


def rmse(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.sqrt(np.square(np.subtract(actual, pred)).mean())


def pearson_corr(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return pearsonr(actual, pred)[0]


train_data = pd.read_csv('./data/Affinity_train_data.csv', sep=',')
# training
train_feature = pd.read_csv('./data/Affinity_feature.csv', sep=',')
train_label = pd.DataFrame(train_data['Label'])
length = int(0.8 * len(train_label))

model = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(8,), random_state=2, alpha=0.001,
                     learning_rate_init=0.05, verbose=True)
# model = xgb.XGBRegressor(max_depth=6, learning_rate=0.01, n_estimators=50)

model.fit(train_feature.iloc[:length, :], train_label.iloc[:length, :].values.ravel())
# model.fit(train_feature.iloc[:, :], train_label.iloc[:, :].values.ravel())
# validation
pred_unlabeled = model.predict(train_feature.iloc[length:, :])
valid_label = train_label['Label'].loc[length:].tolist()
predict = np.round(pred_unlabeled)
rmse_a = rmse(valid_label, predict)
pcc_a = pearson_corr(valid_label, predict)
print(rmse_a, pcc_a)

joblib.dump(model, 'affinity.model')
