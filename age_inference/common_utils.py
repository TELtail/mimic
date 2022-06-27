import numpy as np
import torch
import datetime
import os
from mymodels import *
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



def select_model(model_name,num_axis,hidden_dim,num_layers,sig_length,model_type):
    #モデル選択
    if model_name == "Lstm_net":
        model = Lstm_net(num_axis,hidden_dim,num_layers,model_type)
    if model_name == "Conv1D_net":
        model = Conv1D_net(num_axis,hidden_dim,sig_length,model_type)
    if model_name == "Linear_net":
        model = Linear_net(num_axis,hidden_dim,num_layers,sig_length,model_type)
    
    return model



def set_log_settings(out_path,config_path):
    with open(config_path,"r") as f:
        log_conf = json.load(f)
    log_conf["handlers"]["fileHandler"]["filename"] = os.path.join(out_path,"train_log.txt") #出力ログのpathを指定
    
    global log_config
    log_config = log_conf


def log_start():
    global log_config
    config.dictConfig(log_config)
    logger = getLogger(__name__)
    return logger
