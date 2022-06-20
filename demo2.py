import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

import time


import xgboost as xgb
import lightgbm as lgb
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# MNISTデータセットの事前準備
(x_train, t_train), (x_test, t_test) = mnist.load_data()
# 学習データと検証データに分割
split_ratio = 0.2
x_train, x_validation, t_train, t_validation = train_test_split(x_train, t_train, test_size=split_ratio)
print(x_train.shape)
print(t_train.shape)
print(x_validation.shape)
print(t_validation.shape)

# 平滑化
x_train = x_train.reshape(-1, 784)
x_validation = x_validation.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
# 正規化
x_train = x_train.astype(float) / 255
x_validation = x_validation.astype(float) / 255
x_test = x_test.astype(float) / 255

print(x_train.shape)
print(t_train.shape)
print(x_validation.shape)
print(t_validation.shape)
# データを設定
xgb_train_data = xgb.DMatrix(x_train, label=t_train)
xgb_eval_data = xgb.DMatrix(x_validation, label=t_validation)
xgb_test_data = xgb.DMatrix(x_test, label=t_test)
lgb_train_data = lgb.Dataset(x_train, label=t_train)
lgb_eval_data = lgb.Dataset(x_validation, label=t_validation, reference=lgb_train_data)

# LightGBMモデル構築
start = time.time()
lgb_params = {"task": "train",
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "num_class": 10}
gbm = lgb.train(lgb_params, lgb_train_data,
                valid_sets=lgb_eval_data, 
                num_boost_round=100, 
                early_stopping_rounds=10)
preds = gbm.predict(x_test)
y_pred = []
for x in preds:
    y_pred.append(np.argmax(x))

print("accuracy score: {}".format(accuracy_score(t_test, y_pred)))
print("elapsed time: {}".format(time.time() - start))