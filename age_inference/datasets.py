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


def mk_dataset(data_pickle_path,age_json_path,train_rate,batch_size,need_elements_list,minimum_signal_length,maximum_signal_length,out_path):


    with open(data_pickle_path,"rb") as f:
        data = pickle.load(f) #信号データ
    with open(age_json_path,"r") as g:
        age_map = json.load(g) #年齢の対応データ
    
    merged_data,data_not_have_feature = merging_data(data,age_map,need_elements_list) #信号と年齢データを対応付ける
    data_signals_age,data_not_have_feature = extractioning_signals(merged_data,age_map,need_elements_list,data_not_have_feature,minimum_signal_length) #必要なデータだけ取得

    
    

    delete_data_info(out_path,data_not_have_feature,age_map,len(data),len(data_signals_age)) #使用しなかったデータを集計してjsonファイルに出力する

    data_x = [] #信号
    data_t = [] #年齢(ラベル)

    for key,one_data in data_signals_age.items():
        
        tmp = np.array(one_data["signals"],dtype=np.float32)[:maximum_signal_length]
        ave = np.nanmean(tmp)
        tmp = np.nan_to_num(tmp,nan=ave) #nanを0で置換
        tmp = torch.tensor(tmp) 
        data_x.append(tmp)
        data_t.append([one_data["age"]])

    plot_age_histogram(data_t,out_path)
    data_x = nn.utils.rnn.pad_sequence(data_x,batch_first=True) #足りないデータはゼロ埋め
    data_t = torch.tensor(np.array(data_t),dtype=torch.int64)
    train_indices, test_indices = train_test_split(list(range(len(data_t))),train_size=train_rate,random_state=SEED) #学習データとテストデータを分割

    dataset = torch.utils.data.TensorDataset(data_x,data_t)

    traindataset = torch.torch.utils.data.Subset(dataset,train_indices) #取得したindexをもとに新たにdatasetを作成
    testdataset = torch.torch.utils.data.Subset(dataset,test_indices) #取得したindexをもとに新たにdatasetを作成
    trainloader = torch.utils.data.DataLoader(traindataset,batch_size=batch_size)
    testloader  = torch.utils.data.DataLoader(testdataset,batch_size=1)

    return trainloader,testloader


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


