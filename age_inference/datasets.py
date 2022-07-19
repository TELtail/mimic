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
    data_not_have_feature = {} #削除したデータをカウントするための辞書
    data_not_have_feature["not_have_need_feature"] = [] #必要な要素（特徴）を持っていなかった要素用
    merged_data = {} #信号と年齢データをくっつけた辞書
    for key,value in data.items():
        merged_data[key] = {}
        merged_data[key]["age"] = age_map["data"][key]["age"]
        merged_data[key]["signals"] = value[0]
        merged_data[key]["fields"] = value[1]
        if len(list(set(merged_data[key]["fields"]["sig_name"]) & set(need_elements_list))) != len(need_elements_list): #必要な要素を持っていなかったらデータ削除
            data_not_have_feature["not_have_need_feature"].append([key,age_map["data"][key]["patient_number"]]) #使用しなかったデータの信号番号と患者番号を取得
            del merged_data[key] #必要な要素（特徴）を持っていなかった要素を削除
    
    return merged_data,data_not_have_feature

def extractioning_signals(merged_data,age_map,need_elements_list,data_not_have_feature,minimum_signal_length):
    data_signlas_age = {} #信号と年齢データをくっつけた辞書
    data_not_have_feature["too_short"] = [] #データが短すぎた用
    for key,one_data in merged_data.items():
        indexs = [] #必要な要素のあるインデックスを格納するリスト
        if np.array(one_data["signals"]).shape[0] < minimum_signal_length: #短すぎるデータは削除
            data_not_have_feature["too_short"].append([key,age_map["data"][key]["patient_number"]])
            continue
        for element in need_elements_list:
            indexs.append(one_data["fields"]["sig_name"].index(element)) #必要なindexを取得
        data_signlas_age[key] = {}
        data_signlas_age[key]["signals"] = one_data["signals"][:,indexs] #必要な信号を抽出
        data_signlas_age[key]["age"] = merged_data[key]["age"]

    return data_signlas_age,data_not_have_feature

def nan_data_delete(data_signals_age,age_map,data_not_have_feature):
    data_signals_age_zero_delete = copy.copy(data_signals_age) #戻り値
    data_not_have_feature["too_many_zero"] = [] #削除したデータのカウント用

    for key,one_data in data_signals_age.items():
        zero_num = np.sum(data_signals_age[key]["signals"]==0,axis=0) #ゼロの数を軸ごとに計算
        if max(zero_num)/len(data_signals_age[key]["signals"]) > 0.05: #ゼロ率が高い信号を削除
            data_not_have_feature["too_many_zero"].append([key,age_map["data"][key]["patient_number"]]) #削除したデータのカウント用
            del data_signals_age_zero_delete[key] #削除実行
    return data_signals_age_zero_delete,data_not_have_feature

def mk_dataset(data_pickle_path,age_json_path,need_elements_list,minimum_signal_length,maximum_signal_length,out_path):


    with open(data_pickle_path,"rb") as f:
        data = pickle.load(f) #信号データ
    with open(age_json_path,"r") as g:
        age_map = json.load(g) #年齢の対応データ
    
    merged_data,data_not_have_feature = merging_data(data,age_map,need_elements_list) #信号と年齢データを対応付ける
    data_signals_age,data_not_have_feature = extractioning_signals(merged_data,age_map,need_elements_list,data_not_have_feature,minimum_signal_length) #必要なデータだけ取得
    data_signals_age_zero_delete,data_not_have_feature = nan_data_delete(data_signals_age,age_map,data_not_have_feature) #0が多すぎるデータの削除
    
    delete_data_info(out_path,data_not_have_feature,age_map,len(data),len(data_signals_age_zero_delete)) #使用しなかったデータを集計してjsonファイルに出力する

    data_x = [] #信号
    data_t = [] #年齢(ラベル)

    for key,one_data in data_signals_age_zero_delete.items():
        
        tmp = np.array(one_data["signals"],dtype=np.float32)[:maximum_signal_length] #ndarrayへの変更、指定された長さで削除
        convert_signal = Convert_signal(tmp)
        tmp = convert_signal.zero_to_nan_to_medi() 

        tmp = torch.tensor(tmp) 
        data_x.append(tmp)
        data_t.append([one_data["age"]])
    
    plot_age_histogram(data_t,out_path)
    data_x = nn.utils.rnn.pad_sequence(data_x,batch_first=True) #足りないデータはゼロ埋め
    data_t = torch.tensor(np.array(data_t),dtype=torch.int64)
    return data_x,data_t

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

class Convert_signal:
    def __init__(self,signal):
        self.signal = signal
        self.ave = np.nanmean(self.signal,axis=0)
        self.medi = np.nanmedian(self.signal,axis=0)
    
    def nan_to_ave(self):
        no_nan_signal = np.nan_to_num(self.signal,nan=self.ave)
        return no_nan_signal 
    
    def nan_to_medi(self):
        no_nan_signal = np.nan_to_num(self.signal,nan=self.medi)
        return no_nan_signal 

    def zero_to_nan_to_ave(self):
        self.signal[self.signal==0] = np.nan
        converted_signal = self.nan_to_ave()
        return converted_signal
    
    def zero_to_nan_to_medi(self):
        self.signal[self.signal==0] = np.nan
        converted_signal = self.nan_to_medi()
        return converted_signal
    
    def outlier_to_nan_to_ave(self):
        pass



def delete_data_info(out_path,data_not_have_feature,age_map,before_num,after_num):
    delete_data_json_path = os.path.join(out_path,"delete_data_info.json")

    delete_data = {} 
    delete_data["total"] = {"before":before_num,"after":after_num}
    delete_data["correction"] = age_map["info"]
    delete_data["delete"] = {}

    for key,value in data_not_have_feature.items():
        value = np.array(value)
        tmp = value[:,1] #患者番号だけを抜き出し
        tmp = set(tmp) #重複要素削除

        delete_data["delete"][key] = {"signal_number":len(value),"patient_number":len(tmp)}

    with open(delete_data_json_path,"w") as f:
        json.dump(delete_data,f,indent=4)



def delete_from_pickle_to_dataframe(data_pickle_path):
    if ".csv" in data_pickle_path:
        detailed_data = mk_data_if_dont_have_data_bin(data_pickle_path)
        flag = "row"
    elif ".bin" in data_pickle_path:
        flag = "analysis"
        with open(data_pickle_path,"rb") as f:
            detailed_data = pickle.load(f) #信号データ
    signal_dataframes = {}
    for signal_name,one_data in detailed_data.items():
        one_signals = pd.DataFrame(one_data[0],columns=one_data[1]["sig_name"])
        signal_dataframes[signal_name] = one_signals

    
    return signal_dataframes,flag

def mk_data_if_dont_have_data_bin(csv_path):
    data = {}
    with open(csv_path,"r") as f:
        paths = csv.reader(f)
        for p in paths:
            p = p[0].split(".dat")[0]
            signals,fields = wfdb.rdsamp(p)
            name = os.path.basename(p)
            data[name] = [signals,fields]
    
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
    

    def run_for_analysis(self):
        self.delete_signal_not_have_need_element_at_least()
        self.delete_signal_too_many_zero()
        self.delete_signal_too_short()
        self.convert_zero_to_nan()
        #self.segmentating_signals_that_zero_continous_for_a_long_time()
        self.convert_nan_to_ffill()
        self.extract_need_elements_from_signals()
        self.shorten_long_signals()
    
    def run_for_row(self):
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


def mk_dataset_v2(data_pickle_path,age_json_path,need_elements_list,minimum_signal_length,maximum_signal_length,out_path,model_type,train_rate,splited_one_signal_length):
    
    signal_dataframes,row_analysis_flag = delete_from_pickle_to_dataframe(data_pickle_path)
    convert_cl = Convert_Delete_signal_dataframes(signal_dataframes,need_elements_list,minimum_signal_length,maximum_signal_length)
    if row_analysis_flag == "row":
        convert_cl.run_for_row()
    elif row_analysis_flag == "analysis":
        convert_cl.run_for_analysis()

    with open(age_json_path,"r") as g:
        age_map = json.load(g) #年齢の対応データ
    data_x,data_t = associate_age_signals(convert_cl.signals,age_map,splited_one_signal_length)
    if splited_one_signal_length:
        data_x,data_t,train_indices,test_indices = split_signals(data_x,data_t,train_rate,600)
    else:
        train_indices = None
        test_indices = None
    
    data_x = nn.utils.rnn.pad_sequence(data_x,batch_first=True) #足りないデータはゼロ埋め
    if len(data_x.shape) == 2:
        data_x = data_x.unsqueeze(dim=2)
    data_t = torch.tensor(np.array(data_t),dtype=torch.int64)
    if model_type == "classification":
        data_t = categorize_dataset_for_classification(data_t)

    return data_x,data_t,train_indices,test_indices

