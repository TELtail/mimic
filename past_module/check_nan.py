import numpy as np
import pickle
import pandas as pd
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


def mk_dataset(data_pickle_path,age_pickle_path,need_elements_list,minimum_signal_length):


    with open(data_pickle_path,"rb") as f:
        data = pickle.load(f) #信号データ
    with open(age_pickle_path,"rb") as g:
        age_map = pickle.load(g) #年齢の対応データ
    
    merged_data = merging_data(data,age_map,need_elements_list) #信号と年齢データを対応付ける
    data_signals_age = extractioning_signals(merged_data,need_elements_list) #必要なデータだけ取得

    nan_percents = []
    for key,one_data in data_signals_age.items():
        if np.array(one_data["signals"]).shape[0] < minimum_signal_length: #短すぎるデータは削除
            continue
        tmp = np.array(one_data["signals"])
        nan_percent = check_nan_num(tmp,need_elements_list)
        nan_percents.append(nan_percent)

    plt.figure(figsize=(16,8))
    plt.rcParams["font.size"] = 30
    plt.xlabel("Percentage of nan in HR")
    plt.ylabel("The number of signals")
    plt.hist(nan_percents,bins=500)
    plt.tight_layout()
    plt.show()

    

def check_nan_num(data_ndarray,need_elements_list):
    data_df = pd.DataFrame(
        data=data_ndarray,
        columns=need_elements_list
    )

    nan_percent = (data_df.isnull().sum()["HR"]) / len(data_df)

    return nan_percent

def main():
    data_pickle_path = "./data/data.bin"
    age_pickle_path = "./data/age_data.bin"
    need_elements_list = ['HR', 'RESP', 'SpO2', 'NBPSys', 'NBPDias', 'NBPMean']
    minimum_signal_length = 300

    mk_dataset(data_pickle_path,age_pickle_path,need_elements_list,minimum_signal_length) #データローダー取得


    

if __name__ == "__main__":
    main()