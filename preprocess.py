import pickle
import json
import pandas as pd
import copy
import matplotlib.pyplot as plt


def convert_from_pickle_to_dataframe(data_pickle_path):
    with open(data_pickle_path,"rb") as f:
        detailed_data = pickle.load(f) #信号データ
    
    signal_dataframes = {}

    for signal_name,one_data in detailed_data.items():
        one_signals = pd.DataFrame(one_data[0],columns=one_data[1]["sig_name"])
        signal_dataframes[signal_name] = one_signals

    
    return signal_dataframes


class Convert_signal_dataframes:
    def __init__(self,signal_dataframes):
        self.signals = signal_dataframes
        self.delete_signal_names = []
    
    def delete_signal_not_have_need_element(self):
        converted_signals = copy.copy(self.signals) #一次的に変換後の信号を入れておくdf

        for signal_name,signal in self.signals.items():
            if set(["HR","RESP","SpO2"]) <=  set(list(signal.columns)): #指定要素を含むかどうか
                converted_signals[signal_name] = signal[["HR","RESP","SpO2"]] #指定要素のみ抽出
            else:
                self.delete_signal_names.append(signal_name) #削除した信号番号を保持
                del converted_signals[signal_name] #指定要素を持たない信号を削除
        
        self.signals = converted_signals

    
    def delete_signal_too_many_zero(self):
        converted_signals = copy.copy(self.signals) #一次的に変換後の信号を入れておくdf

        for signal_name,signal in self.signals.items():
            max_zero_num  = max((signal==0).sum())
            if max_zero_num / len(signal) > 0.05:
                self.delete_signal_names.append(signal_name)
                del converted_signals[signal_name]
        
        self.signals = converted_signals


def main():
    data_pickle_path = "./data/data.bin"
    age_json_path = "./data/age_data.json"
    with open(age_json_path,"r") as g:
        age_map = json.load(g) #年齢の対応データ
    
    signal_dataframes = convert_from_pickle_to_dataframe(data_pickle_path)
    convert = Convert_signal_dataframes(signal_dataframes)
    convert.delete_signal_not_have_need_element()
    convert.delete_signal_too_many_zero()
    
    


if __name__ == "__main__":
    main()