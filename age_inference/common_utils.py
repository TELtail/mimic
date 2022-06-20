import numpy as np
import torch
import datetime
import os
import age_inference.mymodels as mymodels
import json
from logging import getLogger,config

SEED = 42

def define_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def mk_out_dir(out_path):
    now = datetime.datetime.now() #現在時刻取得
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    out_path = os.path.join(out_path,now_string)
    os.mkdir(out_path) #保存ディレクトリ作成
    return out_path



def select_model(model_name,num_axis,hidden_dim,num_layers,sig_length):
    #モデル選択
    if model_name == "Lstm_net":
        model = mymodels.Lstm_net(num_axis,hidden_dim,num_layers)
    if model_name == "Conv1D_net":
        model = mymodels.Conv1D_net(num_axis,hidden_dim,sig_length)
    if model_name == "Linear_net":
        model = mymodels.Linear_net(num_axis,hidden_dim,num_layers,sig_length)
    
    return model


def log_start(out_path,config_path):
    with open(config_path,"r") as f:
        log_conf = json.load(f)
    log_conf["handlers"]["fileHandler"]["filename"] = os.path.join(out_path,"train_log.txt") #出力ログのpathを指定
    config.dictConfig(log_conf)
    logger = getLogger(__name__)
