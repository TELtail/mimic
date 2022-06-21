import pickle
import json
import pandas as pd


def need_dataframes_get(data_pickle_path):
    with open(data_pickle_path,"rb") as f:
        detailed_data = pickle.load(f) #信号データ
    
    signal_dataframes = {}
    delete_signal_names = []
    for signal_name,one_data in detailed_data.items():
        one_signals = pd.DataFrame(detailed_data[signal_name][0],columns=detailed_data[signal_name][1]["sig_name"])
        if set(["HR","RESP","SpO2"]) <=  set(one_signals.columns.values):
            signal_dataframes[signal_name] = one_signals[["HR","RESP","SpO2"]]
        else:
            delete_signal_names.append(signal_name)
    
    print(signal_dataframes)
    
    return signal_dataframes,delete_signal_names




def main():
    data_pickle_path = "./data/data.bin"
    age_json_path = "./data/age_data.json"
    with open(age_json_path,"r") as g:
        age_map = json.load(g) #年齢の対応データ
    
    signal_dataframes,delete_signal_names = need_dataframes_get(data_pickle_path)

    
    

if __name__ == "__main__":
    main()