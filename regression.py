import torch
import torch.nn as nn
from IPython.display import display
import scipy
from sklearn.model_selection import train_test_split
import wfdb
import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt


def merging_data(data,age_map,need_elements_list):
    merged_data = {}
    for key,value in data.items():
        merged_data[key] = {}
        merged_data[key]["age"] = age_map[key]
        merged_data[key]["signals"] = value[0]
        merged_data[key]["fields"] = value[1]
        if len(list(set(merged_data[key]["fields"]["sig_name"]) & set(need_elements_list))) != len(need_elements_list): #必要な要素を持っていなかったらデータ削除
            del merged_data[key]
    
    return merged_data

def extractioning_signals(merged_data,need_elements_list):
    data_signlas_age = {}

    for key,one_data in merged_data.items():
        indexs = []
        for element in need_elements_list:
            indexs.append(one_data["fields"]["sig_name"].index(element)) #必要なindexを取得
        data_signlas_age[key] = {}
        data_signlas_age[key]["signals"] = one_data["signals"][:,indexs] #必要な信号を抽出
        data_signlas_age[key]["age"] = merged_data[key]["age"]
    
    return data_signlas_age


def mk_dataset(data_pickle_path,age_pickle_path,train_rate,batch_size,need_elements_list,minimum_signal_length):


    with open(data_pickle_path,"rb") as f:
        data = pickle.load(f) #信号データ
    with open(age_pickle_path,"rb") as g:
        age_map = pickle.load(g) #年齢の対応データ
    
    merged_data = merging_data(data,age_map,need_elements_list) #信号と年齢データを対応付ける
    data_signals_age = extractioning_signals(merged_data,need_elements_list) #必要なデータだけ取得

    data_x = [] #信号
    data_t = [] #年齢(ラベル)
    length_list = []
    for key,one_data in data_signals_age.items():
        tmp = np.array(one_data["signals"],dtype=np.float32)
        length_list.append(tmp.shape[0])
        tmp = np.nan_to_num(tmp,nan=0) #nanを0で置換
        tmp = torch.tensor(tmp) 
        data_x.append(tmp)
        data_t.append([one_data["age"]])

    data_x = nn.utils.rnn.pad_sequence(data_x,batch_first=True) #足りないデータはゼロ埋め
    data_t = torch.tensor(np.array(data_t),dtype=torch.int64)
    ################################################################
    plt.figure(figsize=(16,8))
    plt.hist(length_list,bins=500)
    plt.title("Signal length")
    plt.show()
    ################################################################


def main():
    data_pickle_path = "./data/data.bin"
    age_pickle_path = "./data/age_data.bin"
    train_rate = 0.8
    batch_size = 8
    hidden_dim = 64
    epochs = 100
    lr = 1e-3
    need_elements_list = ['HR', 'RESP', 'SpO2', 'NBPSys', 'NBPDias', 'NBPMean']
    minimum_signal_length = 300

    mk_dataset(data_pickle_path,age_pickle_path,train_rate,batch_size,need_elements_list,minimum_signal_length) #データローダー取得
    
    

    

if __name__ == "__main__":
    main()