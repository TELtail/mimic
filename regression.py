import torch
import torch.nn as nn
from IPython.display import display
import wfdb
import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt

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
    merged_data = {}

    with open(data_pickle_path,"rb") as f:
        data = pickle.load(f)
    with open(age_pickle_path,"rb") as g:
        age_map = pickle.load(g)
    
    for key,value in data.items():
        merged_data[key] = {}
        merged_data[key]["age"] = age_map[key]
        merged_data[key]["signals"] = value[0]
        merged_data[key]["fields"] = value[1]
    
    verificate_elements(merged_data)


def verificate_elements(merged_data):
    aggregate_results = {}
    for key,value in merged_data.items():
        for da in value["fields"]["sig_name"]:
            if da not in aggregate_results.keys():
                aggregate_results[da] = 0
            aggregate_results[da]+=1
    print(len(merged_data))
    print(aggregate_results)

    def plot_bar():
        sorted_results = sorted(aggregate_results.items(),key=lambda x:x[1],reverse=True)
        x = []
        y = []
        for a in sorted_results:
            x.append(a[0])
            y.append(a[1])
        plt.bar(x,y)
        plt.show()
    plot_bar()


def main():
    data_pickle_path = "./data/data.bin"
    age_pickle_path = "./data/age_data.bin"

    mk_dataset(data_pickle_path,age_pickle_path)
    

if __name__ == "__main__":
    main()