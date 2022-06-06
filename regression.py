import torch
import torch.nn as nn
from IPython.display import display
import wfdb
import numpy as np
import glob
import os
import pickle

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        
    def forward(self,x):
        
        return x



def mk_data_pickle(dataset_path):
    data = {}

    file_paths = glob.glob(dataset_path+"/*.hea")
    for path in file_paths:
        name = os.path.basename(path).split(".")[0]
        signals,fields = wfdb.rdsamp(path.split(".")[0])
        data[name] = [signals,fields]
    
    with open("data.bin","wb") as f:
        pickle.dump(data,f)

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


def mk_dataset(data_pickle_path,age_pickle_path):
    need_elements_list = ['HR', 'RESP', 'SpO2', 'NBPSys', 'NBPDias', 'NBPMean']
    min_length = 300

    with open(data_pickle_path,"rb") as f:
        data = pickle.load(f) #信号データ
    with open(age_pickle_path,"rb") as g:
        age_map = pickle.load(g) #年齢の対応データ
    
    merged_data = merging_data(data,age_map,need_elements_list)
    data_signals_age = extractioning_signals(merged_data,need_elements_list)

    data_x = []
    data_t = []

    for key,one_data in data_signals_age.items():
        if np.array(one_data["signals"]).shape[0] < min_length:
            continue
        tmp = np.array(one_data["signals"],dtype=np.float64)
        print(tmp.shape)
        data_x.append(tmp)
        data_t.append(one_data["age"])
    data_x = torch.tensor(np.array(data_x,dtype=np.float64))
    data_t = torch.tensor(np.array(data_t))



def main():
    data_pickle_path = "./data/data.bin"
    age_pickle_path = "./data/age_data.bin"

    mk_dataset(data_pickle_path,age_pickle_path)
    

if __name__ == "__main__":
    main()