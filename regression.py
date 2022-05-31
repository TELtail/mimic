import collections
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



def mk_dataset(dataset_path):
    data = {}

    file_paths = glob.glob(dataset_path+"/*.hea")
    for path in file_paths:
        name = os.path.basename(path).split(".")[0]
        signals,fields = wfdb.rdsamp(path.split(".")[0])
        data[name] = [signals,fields]
    
    with open("data.bin","wb") as f:
        pickle.dump(data,f)
    


def main():
    
    dataset_path = "data\mimic-II"
    #mk_dataset(dataset_path)

if __name__ == "__main__":
    main()