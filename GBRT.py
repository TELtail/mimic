import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
import lightgbm as lgb
import wfdb
import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt
import datetime
import argparse
from logging import getLogger,config
import json
SEED = 42

def define_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


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


def mk_dataset(data_pickle_path,age_json_path,train_rate,need_elements_list,minimum_signal_length,maximum_signal_length,out_path):


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
        data_x.append(tmp)
        data_t.append(one_data["age"])
    
    data_x = np.array(data_x)
    data_t = np.array(data_t)


    plot_age_histogram(data_t,out_path)
    x_train, x_test, t_train, t_test = train_test_split(data_x,data_t,train_size=train_rate,random_state=SEED) #学習データとテストデータを分割
    x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])

    return x_train, x_test, t_train, t_test


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

def plot_age_histogram(data_t,out_path):
    #年齢の分布をプロット
    labels = np.array(data_t)
    labels = np.ravel(labels)
    hist_png_path = os.path.join(out_path,"age_hist.png")
    fig_age = plt.figure(figsize=(12,8))
    ax_age = fig_age.add_subplot(111)
    ax_age.hist(labels,bins=70)
    ax_age.set_xlabel("Age")
    ax_age.set_ylabel("Number of people")
    plt.rcParams["font.size"] = 30
    fig_age.savefig(hist_png_path)

    #一様分布、正規分布でのMSEの比較
    uniform = np.random.randint(20,90,len(labels))
    normal = np.random.normal(70,20,len(labels))
    all = np.full(len(labels),70)
    mse_uniform = mean_squared_error(labels,uniform)
    mse_normal = mean_squared_error(labels,normal)
    mse_all = mean_squared_error(labels,all)
    logger.info("All 70: {}".format(mse_all))
    logger.info("Uniform distribution   age U(20,90): {}".format(mse_uniform))
    logger.info("Normal distribution    age N(70,20^2): {}".format(mse_normal))
    logger.info("---------------------------------")

def plot_result(t_test,t_pred,out_path):
    result_fig = plt.figure(figsize=(12,9))
    result_ax = result_fig.add_subplot(111)
    result_ax.plot(t_test, t_test, color = 'red', label = 'x=y') # 直線y = x (真値と予測値が同じ場合は直線状に点がプロットされる)
    result_ax.scatter(t_test, t_pred) # 散布図のプロット
    plt.rcParams["font.size"] = 30
    result_ax.set_xlabel('Correct Answer Label') # x軸ラベル
    result_ax.set_ylabel('Predicted Label') # y軸ラベル
    plt.savefig(os.path.join(out_path,"predict_result.png"))



def mk_out_dir(out_path):
    now = datetime.datetime.now() #現在時刻取得
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    out_path = os.path.join(out_path,now_string)
    os.mkdir(out_path) #保存ディレクトリ作成
    return out_path

def log_start(out_path,config_path):
    with open(config_path,"r") as f:
        log_conf = json.load(f)
    log_conf["handlers"]["fileHandler"]["filename"] = os.path.join(out_path,"train_log.txt") #出力ログのpathを指定
    config.dictConfig(log_conf)

    global logger #loggerをグローバル変数として定義
    logger = getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser("MIMIC-IIデータセットで年齢の学習、推論を行うプログラム")
    parser.add_argument("--data_path",help="信号のバイナリデータのパス")
    parser.add_argument("--age_path",help="年齢のバイナリデータのパス")
    parser.add_argument("--out_path",help="グラフ等を出力するパス",default="./out")
    parser.add_argument("--train_rate",help="学習データの割合",type=float,default=0.8)
    parser.add_argument("--min",help="最小の信号の長さ",type=int,default=300)
    parser.add_argument("--max",help="最大の信号の長さ",type=int,default=1500)
    parser.add_argument("--need_elements",help="必要な要素名",nargs="*",default=['HR', 'RESP', 'SpO2'])
    parser.add_argument("--config",help="log_config.jsonのpath指定",default="./log_config.json")

    args = parser.parse_args()

    data_pickle_path = args.data_path
    age_json_path = args.age_path
    out_path = args.out_path
    train_rate = args.train_rate
    minimum_signal_length = args.min
    maximum_signal_length = args.max
    need_elements_list = args.need_elements
    config_path = args.config


    return data_pickle_path,age_json_path,out_path,train_rate,minimum_signal_length,maximum_signal_length,need_elements_list,config_path
def main():
    data_pickle_path,age_json_path,out_path,train_rate,minimum_signal_length,maximum_signal_length,need_elements_list,config_path = get_parser()
    out_path = mk_out_dir(out_path)
    log_start(out_path,config_path)

    define_seed() #seed固定
    x_train, x_test, t_train, t_test= mk_dataset(data_pickle_path,age_json_path,train_rate,need_elements_list,minimum_signal_length,maximum_signal_length,out_path) #データローダー取得

    # モデルの学習
    model = lgb.LGBMRegressor() # モデルのインスタンスの作成
    model.fit(x_train, t_train) # モデルの学習

    # テストデータの予測
    t_pred = model.predict(x_test)
    mseloss = (mean_squared_error(t_test, t_pred))
    logger.info("MSEloss:"+str(mseloss))
    plot_result(t_test,t_pred,out_path)

    

if __name__ == "__main__":
    main()