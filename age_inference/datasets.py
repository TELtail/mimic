import glob
import os
import pickle
import numpy as np
import json
import torch
import torch.nn as nn
import wfdb
from plot_glaph import plot_age_histogram
from common_utils import SEED
from sklearn.model_selection import train_test_split
import copy
import pandas as pd
import matplotlib.pyplot as plt
import csv

def get_loader(data_x,data_t,train_rate,batch_size,train_indices,test_indices):
    #input data_x (tensor)
    #      data_t (tensor)
    if train_indices == None and test_indices == None:
        train_indices, test_indices = train_test_split(list(range(len(data_t))),train_size=train_rate,random_state=SEED) #学習データとテストデータを分割

    dataset = torch.utils.data.TensorDataset(data_x,data_t)

    traindataset = torch.torch.utils.data.Subset(dataset,train_indices) #取得したindexをもとに新たにdatasetを作成
    testdataset = torch.torch.utils.data.Subset(dataset,test_indices) #取得したindexをもとに新たにdatasetを作成
    trainloader = torch.utils.data.DataLoader(traindataset,batch_size=batch_size)
    testloader  = torch.utils.data.DataLoader(testdataset,batch_size=1)

    return trainloader,testloader


def delete_from_pickle_to_dataframe(data_pickle_path):
    if ".csv" in data_pickle_path:
        detailed_data = mk_data_if_dont_have_data_bin(data_pickle_path)
    elif ".bin" in data_pickle_path:
        with open(data_pickle_path,"rb") as f:
            detailed_data = pickle.load(f) #信号データ
    signal_dataframes = {}
    for signal_name,one_data in detailed_data.items():
        one_signals = pd.DataFrame(one_data[0],columns=one_data[1]["sig_name"])
        signal_dataframes[signal_name] = one_signals

    
    return signal_dataframes

def mk_data_if_dont_have_data_bin(csv_path):
    data = {}
    with open(csv_path,"r") as f:
        paths = csv.reader(f)
        for p in paths:
            p = p[0]
            dir_name = os.path.dirname(p)
            base_name = os.path.basename(p).split(".")[0]

            signals,fields = wfdb.rdsamp(os.path.join(dir_name,base_name))
            data[base_name] = [signals,fields]
    
    return data

class Convert_Delete_signal_dataframes:
    def __init__(self,signal_dataframes,need_elements_list,minimum_signal_length,maximum_signal_length):
        self.signals = signal_dataframes
        self.delete_signal_names = {}
        self.need_elements_list = need_elements_list
        self.minimum_signal_length = minimum_signal_length
        self.maximum_signal_length = maximum_signal_length
    
    def delete_signal_not_have_need_element_at_least(self):
        deleteed_signals = copy.copy(self.signals) #一次的に変換後の信号を入れておくdf
        self.delete_signal_names["not_have_need_element_at_least"] = []

        for signal_name,signal in self.signals.items():
            if set(self.need_elements_list) <=  set(list(signal.columns)): #指定要素を含むかどうか
                deleteed_signals[signal_name] = signal[self.need_elements_list] #指定要素のみ抽出
            else:
                self.delete_signal_names["not_have_need_element_at_least"].append(signal_name) #削除した信号番号を保持
                del deleteed_signals[signal_name] #指定要素を持たない信号を削除
        
        self.signals = deleteed_signals
    
    def delete_signal_not_have_need_any_element(self):
        deleteed_signals = copy.copy(self.signals) #一次的に変換後の信号を入れておくdf
        self.delete_signal_names["not_have_need_element_at_least"] = []

        for signal_name,signal in self.signals.items():
            exist_elements_list = [i for i in signal.columns if i in self.need_elements_list] #指定要素名を取得
            if len(exist_elements_list) != 0: #指定要素を一つでも含んでいれば
                deleteed_signals[signal_name] = signal[exist_elements_list] #指定要素のみ抽出
            else:
                self.delete_signal_names["not_have_need_element_at_least"].append(signal_name) #削除した信号番号を保持
                del deleteed_signals[signal_name] #指定要素を持たない信号を削除
        
        self.signals = deleteed_signals

    def delete_signal_too_many_zero(self):
        deleteed_signals = copy.copy(self.signals) #一次的に変換後の信号を入れておくdf
        self.delete_signal_names["too_many_zero"] = []

        for signal_name,signal in self.signals.items():
            max_zero_num  = max((signal==0).sum())
            if max_zero_num / len(signal) > 0.05:
                self.delete_signal_names["too_many_zero"].append(signal_name)
                del deleteed_signals[signal_name]
        
        self.signals = deleteed_signals
    
    def delete_signal_too_short(self):
        deleteed_signals = copy.copy(self.signals) #一次的に変換後の信号を入れておくdf
        self.delete_signal_names["too_short"] = []

        for signal_name,signal in self.signals.items():
            if len(signal) < self.minimum_signal_length:
                self.delete_signal_names["too_short"].append(signal_name)
                del deleteed_signals[signal_name]
        
        self.signals = deleteed_signals
    
    def convert_nan_to_ffill(self):
        for signal_name,signal in self.signals.items():
            self.signals[signal_name] = signal.fillna(method="ffill")
            if np.isnan(np.array(self.signals[signal_name])).sum() != 0:
                self.signals[signal_name] = self.signals[signal_name].fillna(method="bfill")
    
    def convert_zero_to_nan(self):
        for signal_name,signal in self.signals.items():
            signal = signal.where(signal>1,np.nan)
            self.signals[signal_name] = signal
    
    def segmentating_signals_that_zero_continous_for_a_long_time(self):
        for signal_name,signal in self.signals.items():
            signal = np.array(signal)
            isnan_tuple = np.where(np.isnan(signal)) #np.nanを持つインデックスを取得
            indexes = []
            for i,j in zip(isnan_tuple[0],isnan_tuple[1]):
                indexes.append([i,j]) 
            
            continue_indexes = []
            i=0
            while(i<len(indexes)):
                continue_times=0
                while([indexes[i][0]+continue_times,indexes[i][1]] in indexes):
                    continue_times+=1
                continue_indexes.append([indexes[i][0],continue_times])
                i+=continue_times
    
    def extract_need_elements_from_signals(self):
        for signal_name,signal in self.signals.items():
            self.signals[signal_name] = signal[self.need_elements_list]
    
    def select_hr_signal(self):
        for signal_name,signal in self.signals.items():
            exist_elements_list = [i for i in signal.columns if i in self.need_elements_list] #指定要素名を取得
            self.signals[signal_name] = signal[exist_elements_list[-1]]
    
    def shorten_long_signals(self):
        for signal_name,signal in self.signals.items():
            if type(signal) == pd.core.frame.DataFrame:
                self.signals[signal_name] = signal.iloc[:self.maximum_signal_length,:]
            elif type(signal) == pd.core.series.Series:
                self.signals[signal_name] = signal.iloc[:self.maximum_signal_length]
    

    def run_using_all_elements(self):
        self.delete_signal_not_have_need_element_at_least()
        self.delete_signal_too_many_zero()
        self.delete_signal_too_short()
        self.convert_zero_to_nan()
        #self.segmentating_signals_that_zero_continous_for_a_long_time()
        self.convert_nan_to_ffill()
        self.extract_need_elements_from_signals()
        self.shorten_long_signals()
    
    def run_using_some_elements(self):
        self.delete_signal_not_have_need_any_element()
        self.delete_signal_too_short()
        self.convert_nan_to_ffill()
        self.select_hr_signal()
        self.shorten_long_signals()

def associate_age_signals(signals,age_map,splited_one_signal_length):
    data_x = []
    data_t = []

    for sig_name,sig in signals.items():
        if splited_one_signal_length:
            data_x.append(np.array(sig.values))
        else:
            data_x.append(torch.tensor(sig.values,dtype=torch.float32))
        if "n" in sig_name:
            data_t.append([age_map["data"][sig_name]["age"]])
        else:
            sig_name = sig_name.split("_")[0] + "n"
            data_t.append([age_map["data"][sig_name]["age"]])
    return data_x,data_t

def categorize_dataset_for_classification(data_t):
    med = torch.median(data_t[:,0])
    for i,age in enumerate(data_t):
        if age[0] < med:
            data_t[i][0] = 0
        else:
            data_t[i][0] = 1
    
    return data_t

def categorize_dataset_for_classification_center_near_deletion(data_x,data_t):

    small_line = torch.quantile(data_t,0.25,interpolation="nearest").item()
    big_line = torch.quantile(data_t,0.75,interpolation="nearest").item()
    down_con,up_con = data_t<small_line, big_line<data_t

    new_data_x = []

    for i in range(len(data_x)):
        if down_con[i] or up_con[i]:
            new_data_x.append(data_x[i].tolist())

    new_data_x = torch.tensor(new_data_x)
    new_data_t = torch.cat((data_t[down_con],data_t[up_con]),0)
    new_data_t[new_data_t<small_line] = 0
    new_data_t[big_line<new_data_t] = 1
    return new_data_x,new_data_t


def split_signals(data_x,data_t,train_rate,splited_one_signal_length): #信号分割を行う
    train_indices,test_indices = train_test_split(list(range(len(data_t))),train_size=train_rate,random_state=SEED) #学習データとテストデータを分割
    new_data_x = []
    new_data_t = []
    for idx in train_indices:
        for start in range(0,len(data_x[idx]),splited_one_signal_length):
            new_data_x.append(torch.tensor(data_x[idx][start:start+splited_one_signal_length],dtype=torch.float32)) #splited_one_signal_lengthの数ごとに分割
            new_data_t.append(data_t[idx]) #対応するラベル取得
    
    train_size = len(new_data_t)
    new_train_indices = range(0,train_size)
    
    for idx in test_indices:
        for start in range(0,len(data_x[idx]),splited_one_signal_length):
            new_data_x.append(torch.tensor(data_x[idx][start:start+splited_one_signal_length],dtype=torch.float32)) #splited_one_signal_lengthの数ごとに分割
            new_data_t.append(data_t[idx]) #対応するラベル取得
    
    new_test_indices = range(train_size,len(new_data_t))

    return new_data_x,new_data_t,new_train_indices,new_test_indices


def mk_dataset_v2(data_pickle_path,age_json_path,need_elements_list,minimum_signal_length,maximum_signal_length,out_path,model_type,train_rate,splited_one_signal_length,use_not_all_elements):
    
    signal_dataframes = delete_from_pickle_to_dataframe(data_pickle_path)
    convert_cl = Convert_Delete_signal_dataframes(signal_dataframes,need_elements_list,minimum_signal_length,maximum_signal_length)
    if use_not_all_elements == True:
        convert_cl.run_using_some_elements()
    else:
        convert_cl.run_using_all_elements()

    with open(age_json_path,"r") as g:
        age_map = json.load(g) #年齢の対応データ
    data_x,data_t = associate_age_signals(convert_cl.signals,age_map,splited_one_signal_length)
    if splited_one_signal_length:
        data_x,data_t,train_indices,test_indices = split_signals(data_x,data_t,train_rate,splited_one_signal_length)
    else:
        train_indices = None
        test_indices = None
    
    data_x = nn.utils.rnn.pad_sequence(data_x,batch_first=True) #足りないデータはゼロ埋め
    if len(data_x.shape) == 2:
        data_x = data_x.unsqueeze(dim=2)
    data_t = torch.tensor(np.array(data_t),dtype=torch.float32)
    if model_type == "classification":
        #data_t = categorize_dataset_for_classification(data_t)
        data_x,data_t = categorize_dataset_for_classification_center_near_deletion(data_x,data_t)
    
    num_axis = data_x.shape[-1]
    
    return data_x,data_t,train_indices,test_indices,num_axis

