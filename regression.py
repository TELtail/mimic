import torch
import torch.nn as nn
from IPython.display import display
import scipy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import wfdb
import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt
import datetime
import shutil
import argparse
from logging import getLogger,config
import logging
import json
SEED = 42

def define_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

class Net(nn.Module):
    def __init__(self,num_axis,hidden_dim):
        super(Net,self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(num_axis,hidden_dim,3,batch_first=True)
        self.fc = nn.Linear(hidden_dim,1)

        
    def forward(self,x):
        _,x = self.lstm(x)
        x = x[0][-1].view(-1, self.hidden_dim)
        x = self.fc(x)
        
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
        tmp = np.nan_to_num(tmp,nan=0) #nanを0で置換
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

def plot_age_histogram(data_t,out_path):
    #年齢の分布をプロット
    labels = np.array(data_t)
    labels = np.ravel(labels)
    hist_png_path = os.path.join(out_path,"age_hist.png")
    plt.figure(figsize=(12,8))
    plt.hist(labels,bins=70)
    plt.xlabel("Age")
    plt.ylabel("Number of people")
    plt.rcParams["font.size"] = 30
    plt.savefig(hist_png_path)

    #一様分布、正規分布でのMSEの比較
    uniform = np.random.randint(20,90,len(labels))
    normal = np.random.normal(70,20,len(labels))
    mse_uniform = mean_squared_error(labels,uniform)
    mse_normal = mean_squared_error(labels,normal)
    logger.info("Uniform distribution   age U(20,90):{}".format(mse_uniform))
    logger.info("Normal distribution    age N(70,20^2):{}".format(mse_normal))
    logger.info("---------------------------------")







def train_method(trainloader,net,optimizer,loss_fn,device,batch_size):
    running_loss = 0
    size = len(trainloader.dataset)
    for i,(inputs,labels) in enumerate(trainloader):
        optimizer.zero_grad() #勾配初期化
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = net(inputs)
        loss = loss_fn(outputs,labels.float())
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if i%10 == 0:
            logger.info(f" {i}/{int(size/batch_size)} loss:{loss}")
    
    running_loss /= (i+1)
    logger.info(f"train_loss:{running_loss}")

    return running_loss

def test_method(testloader,net,optimizer,loss_fn,device):
    running_loss = 0
    size = len(testloader.dataset)
    for i,(inputs,labels) in enumerate(testloader):
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = net(inputs)
        loss = loss_fn(outputs,labels.float())
        running_loss += loss.item()
    
    running_loss /= (i+1)
    logger.info(f"test_loss:{running_loss}")

    return running_loss



def mk_out_dir(out_path):
    now = datetime.datetime.now() #現在時刻取得
    now_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    out_path = os.path.join(out_path,now_string)
    os.mkdir(out_path) #保存ディレクトリ作成
    return out_path

def plot_loss_glaph(epoch_loss,out_path):
    labels = ["train","test"]
    epoch_loss = np.array(epoch_loss) #スライスできるようにndarrayに変更
    for i in range(2): #学習データとテストデータのlossだから2
        plt.plot(range(len(epoch_loss)),epoch_loss[:,i],label=labels[i])
    png_path = os.path.join(out_path,"loss.png")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.rcParams["font.size"] = 20
    plt.tight_layout()
    plt.legend()
    plt.savefig(png_path)

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
    parser.add_argument("--batch_size",help="バッチサイズ",type=int,default=8)
    parser.add_argument("--hidden_dim",help="LSTMの次元",type=int,default=64)
    parser.add_argument("--epochs",help="epoch数",type=int,default=100)
    parser.add_argument("--lr",help="学習率",type=float,default=1e-3)
    parser.add_argument("--min",help="最小の信号の長さ",type=int,default=300)
    parser.add_argument("--max",help="最大の信号の長さ",type=int,default=1500)
    parser.add_argument("--need_elements",help="必要な要素名",nargs="*",default=['HR', 'RESP', 'SpO2'])
    parser.add_argument("--config",help="log_config.jsonのpath指定",default="./log_config.json")

    args = parser.parse_args()

    data_pickle_path = args.data_path
    age_json_path = args.age_path
    out_path = args.out_path
    train_rate = args.train_rate
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    epochs = args.epochs
    lr = args.lr
    minimum_signal_length = args.min
    maximum_signal_length = args.max
    need_elements_list = args.need_elements
    config_path = args.config


    return data_pickle_path,age_json_path,out_path,train_rate,batch_size,hidden_dim,epochs,lr,minimum_signal_length,maximum_signal_length,need_elements_list,config_path

def print_parser(data_pickle_path,age_json_path,out_path,train_rate,batch_size,hidden_dim,epochs,lr,minimum_signal_length,maximum_signal_length,need_elements_list):
    logger.info("---------------------------------")
    logger.info("data_pickle_path:{}".format(data_pickle_path))
    logger.info("age_json_path:{}".format(age_json_path))
    logger.info("out_path:{}".format(out_path))
    logger.info("train_rate:{}".format(train_rate))
    logger.info("batch_size:{}".format(batch_size))
    logger.info("hidden_dim:{}".format(hidden_dim))
    logger.info("epochs:{}".format(epochs))
    logger.info("lr:{}".format(lr))
    logger.info("minimum_signal_length:{}".format(minimum_signal_length))
    logger.info("maximum_signal_length:{}".format(maximum_signal_length))
    logger.info("need_elements_list:{}".format(need_elements_list))
    logger.info("---------------------------------")

def main():
    data_pickle_path,age_json_path,out_path,train_rate,batch_size,hidden_dim,epochs,lr,minimum_signal_length,maximum_signal_length,need_elements_list,config_path = get_parser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = mk_out_dir(out_path)
    log_start(out_path,config_path)

    logger.info("Device:{}".format(device))


    print_parser(data_pickle_path,age_json_path,out_path,train_rate,batch_size,hidden_dim,epochs,lr,minimum_signal_length,maximum_signal_length,need_elements_list)

    
    define_seed() #seed固定
    trainloader,testloader = mk_dataset(data_pickle_path,age_json_path,train_rate,batch_size,need_elements_list,minimum_signal_length,maximum_signal_length,out_path) #データローダー取得
    num_axis = len(need_elements_list)
    net = Net(num_axis,hidden_dim).to(device)
    logger.info(net)


    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    loss_fn = nn.MSELoss()
    epoch_loss = [] #グラフに出力するための損失格納用リスト
    try:
        for epoch in range(epochs):
            logger.info(f"----- epoch:{epoch+1} ---------------------------")
            train_running_loss = train_method(trainloader,net,optimizer,loss_fn,device,batch_size)
            test_running_loss = test_method(testloader,net,optimizer,loss_fn,device)
            epoch_loss.append([train_running_loss,test_running_loss])
    except KeyboardInterrupt:
        pass

    if len(epoch_loss) != 0:
        plot_loss_glaph(epoch_loss,out_path) #1エポックでもあれば損失グラフ生成
    else:
        logging.shutdown()
        shutil.rmtree(out_path) #1エポックもなければディレクトリごと削除


if __name__ == "__main__":
    main()