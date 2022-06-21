import math
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import json


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


def mk_dataset(data_pickle_path,age_json_path,minimum_signal_length,maximum_signal_length,need_elements_list,plot_num):


    with open(data_pickle_path,"rb") as f:
        data = pickle.load(f) #信号データ
    with open(age_json_path,"r") as g:
        age_map = json.load(g) #年齢の対応データ
    
    merged_data,data_not_have_feature = merging_data(data,age_map,need_elements_list) #信号と年齢データを対応付ける
    data_signals_age,data_not_have_feature = extractioning_signals(merged_data,age_map,need_elements_list,data_not_have_feature,minimum_signal_length) #必要なデータだけ取得

    data_x = [] #信号
    data_t = [] #年齢(ラベル)

    k = 0
    plot_num_sqrt = int(math.sqrt(plot_num))
    for key,one_data in data_signals_age.items():
        tmp = np.array(one_data["signals"],dtype=np.float32)

        if k % plot_num == 0: 
            fig,axes = plt.subplots(plot_num_sqrt,2*plot_num_sqrt,figsize=(16,8))
            axes = np.ravel(axes)
            j=0
        axes[j].plot(range(len(tmp)),tmp[:,0:3])
        j+=1
        
        convert_signal = Convert_signal(tmp)
        tmp = convert_signal.zero_to_nan_to_medi()
        axes[j].plot(range(len(tmp)),tmp[:,0:3])
        j+=1

        if (k+1) % plot_num == 0:
            plt.show()
        k += 1

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


def get_parser():
    parser = argparse.ArgumentParser("MIMIC-IIデータセットで年齢の学習、推論を行うプログラム")
    parser.add_argument("--data_path",help="信号のバイナリデータのパス")
    parser.add_argument("--age_path",help="年齢のバイナリデータのパス")
    parser.add_argument("--min",help="最小の信号の長さ",type=int,default=300)
    parser.add_argument("--max",help="最大の信号の長さ",type=int,default=1500)
    parser.add_argument("--need_elements",help="必要な要素名",nargs="*",default=['HR', 'RESP', 'SpO2'])
    parser.add_argument("--plot_num",help="一度にプロットする数",type=int,default=9)


    args = parser.parse_args()
    data_pickle_path = args.data_path
    age_json_path = args.age_path
    minimum_signal_length = args.min
    maximum_signal_length = args.max
    need_elements_list = args.need_elements
    plot_num = args.plot_num


    return data_pickle_path,age_json_path,minimum_signal_length,maximum_signal_length,need_elements_list,plot_num

def main():
    data_pickle_path,age_json_path,minimum_signal_length,maximum_signal_length,need_elements_list,plot_num = get_parser()

    mk_dataset(data_pickle_path,age_json_path,minimum_signal_length,maximum_signal_length,need_elements_list,plot_num) #データローダー取得



if __name__ == "__main__":
    main()