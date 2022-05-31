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


def mk_dataset(data_pickle_path,age_pickle_path):
    need_elements_list = ['HR', 'RESP', 'SpO2', 'NBPSys', 'NBPDias', 'NBPMean']

    merged_data = {}

    with open(data_pickle_path,"rb") as f:
        data = pickle.load(f)
    with open(age_pickle_path,"rb") as g:
        age_map = pickle.load(g)
    i=0
    for key,value in data.items():
        merged_data[key] = {}
        merged_data[key]["age"] = age_map[key]
        merged_data[key]["signals"] = value[0]
        merged_data[key]["fields"] = value[1]
        if len(list(set(merged_data[key]["fields"]["sig_name"]) & set(need_elements_list))) == 6:
            i += 1
            print(i,merged_data[key]["fields"]["sig_name"])



def main():
    data_pickle_path = "./data/data.bin"
    age_pickle_path = "./data/age_data.bin"

    mk_dataset(data_pickle_path,age_pickle_path)
    

if __name__ == "__main__":
    main()