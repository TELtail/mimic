import torch
import torch.nn as nn
from IPython.display import display
import scipy
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


def mk_dataset(data_pickle_path,age_pickle_path,train_rate,batch_size,need_elements_list,minimum_signal_length,maximum_signal_length):


    with open(data_pickle_path,"rb") as f:
        data = pickle.load(f) #信号データ
    with open(age_pickle_path,"rb") as g:
        age_map = pickle.load(g) #年齢の対応データ
    
    merged_data = merging_data(data,age_map,need_elements_list) #信号と年齢データを対応付ける
    data_signals_age = extractioning_signals(merged_data,need_elements_list) #必要なデータだけ取得

    data_x = [] #信号
    data_t = [] #年齢(ラベル)

    for key,one_data in data_signals_age.items():
        if np.array(one_data["signals"]).shape[0] < minimum_signal_length: #短すぎるデータは削除
            continue
        tmp = np.array(one_data["signals"],dtype=np.float32)[:maximum_signal_length]
        tmp = np.nan_to_num(tmp,nan=0) #nanを0で置換
        tmp = torch.tensor(tmp) 
        data_x.append(tmp)
        data_t.append([one_data["age"]])

    data_x = nn.utils.rnn.pad_sequence(data_x,batch_first=True) #足りないデータはゼロ埋め
    data_t = torch.tensor(np.array(data_t),dtype=torch.int64)
    train_indices, test_indices = train_test_split(list(range(len(data_t))),train_size=train_rate,random_state=SEED) #学習データとテストデータを分割

    dataset = torch.utils.data.TensorDataset(data_x,data_t)

    traindataset = torch.torch.utils.data.Subset(dataset,train_indices) #取得したindexをもとに新たにdatasetを作成
    testdataset = torch.torch.utils.data.Subset(dataset,test_indices) #取得したindexをもとに新たにdatasetを作成
    trainloader = torch.utils.data.DataLoader(traindataset,batch_size=batch_size)
    testloader  = torch.utils.data.DataLoader(testdataset,batch_size=1)

    return trainloader,testloader

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
            print(f" {i}/{int(size/batch_size)} loss:{loss}")
    
    running_loss /= (i+1)
    print(f"train_loss:{running_loss}")

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
    print(f"test_loss:{running_loss}")

    return running_loss



def main():
    data_pickle_path = "./data/data.bin"
    age_pickle_path = "./data/age_data.bin"
    train_rate = 0.8
    batch_size = 8
    hidden_dim = 64
    epochs = 100
    lr = 1e-3
    need_elements_list = ['HR', 'RESP', 'SpO2']
    minimum_signal_length = 300
    maximum_signal_length = 1500

    device = "cuda" if torch.cuda.is_available() else "cpu"
    define_seed() #seed固定
    trainloader,testloader = mk_dataset(data_pickle_path,age_pickle_path,train_rate,batch_size,need_elements_list,minimum_signal_length,maximum_signal_length) #データローダー取得
    num_axis = len(need_elements_list)
    net = Net(num_axis,hidden_dim).to(device)


    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    loss_fn = nn.MSELoss()
    epoch_loss = []
    for epoch in range(epochs):
        print(f"----- epoch:{epoch+1} ---------------------------")
        train_running_loss = train_method(trainloader,net,optimizer,loss_fn,device,batch_size)
        test_running_loss = test_method(testloader,net,optimizer,loss_fn,device)
        epoch_loss.append([train_running_loss,test_running_loss])
        

    

if __name__ == "__main__":
    main()