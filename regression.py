import torch
import torch.nn as nn
from IPython.display import display
from sklearn.model_selection import train_test_split
import wfdb
import numpy as np
import glob
import os
import pickle
SEED = 42

def define_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

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


def mk_dataset(data_pickle_path,age_pickle_path,train_rate,batch_size):
    need_elements_list = ['HR', 'RESP', 'SpO2', 'NBPSys', 'NBPDias', 'NBPMean']
    min_length = 300

    with open(data_pickle_path,"rb") as f:
        data = pickle.load(f) #信号データ
    with open(age_pickle_path,"rb") as g:
        age_map = pickle.load(g) #年齢の対応データ
    
    merged_data = merging_data(data,age_map,need_elements_list) #信号と年齢データを対応付ける
    data_signals_age = extractioning_signals(merged_data,need_elements_list) #必要なデータだけ取得

    data_x = [] #信号
    data_t = [] #年齢(ラベル)

    for key,one_data in data_signals_age.items():
        if np.array(one_data["signals"]).shape[0] < min_length: #短すぎるデータは削除
            continue
        tmp = torch.tensor(np.array(one_data["signals"],dtype=np.float64)) 
        data_x.append(tmp)
        data_t.append(one_data["age"])
        print(tmp.size())

    data_x = nn.utils.rnn.pad_sequence(data_x,batch_first=True) #足りないデータはゼロ埋め
    data_t = torch.tensor(np.array(data_t))
    train_indices, test_indices = train_test_split(list(range(len(data_t))),train_size=train_rate,random_state=SEED) #学習データとテストデータを分割

    dataset = torch.utils.data.TensorDataset(data_x,data_t)

    traindataset = torch.torch.utils.data.Subset(dataset,train_indices) #取得したindexをもとに新たにdatasetを作成
    testdataset = torch.torch.utils.data.Subset(dataset,test_indices) #取得したindexをもとに新たにdatasetを作成
    trainloader = torch.utils.data.DataLoader(traindataset,batch_size=batch_size)
    testloader  = torch.utils.data.DataLoader(testdataset,batch_size=batch_size)

    return trainloader,testloader



def main():
    data_pickle_path = "./data/data.bin"
    age_pickle_path = "./data/age_data.bin"
    train_rate = 0.8
    batch_size = 32

    define_seed()
    trainloader,testloader = mk_dataset(data_pickle_path,age_pickle_path,train_rate,batch_size)

    

if __name__ == "__main__":
    main()