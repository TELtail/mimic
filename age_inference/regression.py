import torch
import torch.nn as nn
import numpy as np
import shutil
from logging import getLogger,config
import logging
from common_utils import mk_out_dir,select_model,set_log_settings,define_seed,log_start
from opts import print_parser,get_parser
from datasets import mk_dataset,get_loader
from plot_glaph import plot_loss_glaph,plot_inference_result

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

def test_method(testloader,net,loss_fn,device,print_result_flag):
    running_loss = 0
    predicted_for_plot = []
    for i,(inputs,labels) in enumerate(testloader):
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = net(inputs)
        if print_result_flag:
            logger.info(outputs)
        loss = loss_fn(outputs,labels.float())
        outputs_np = outputs.to('cpu').detach().numpy().copy().flatten()[0] #プロット用に、ndarray → 一次元化
        labels_np = labels.to('cpu').detach().numpy().copy().flatten()[0] #プロット用に、ndarray → 一次元化
        predicted_for_plot.append([outputs_np,labels_np])
        running_loss += loss.item()
    
    running_loss /= (i+1)
    logger.info(f"test_loss:{running_loss}")

    return running_loss,np.array(predicted_for_plot)



def main():
    (data_pickle_path,age_json_path,out_path,train_rate,
    batch_size,hidden_dim,num_layers,epochs,lr,minimum_signal_length,
    maximum_signal_length,need_elements_list,config_path,print_result_flag,
    model_name) = get_parser()
    out_path = mk_out_dir(out_path)
    global logger
    set_log_settings(out_path,config_path)
    logger = log_start()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device:{}".format(device))
    
    

    print_parser(data_pickle_path,age_json_path,out_path,train_rate,batch_size,hidden_dim,num_layers,epochs,lr,minimum_signal_length,maximum_signal_length,need_elements_list,config_path,print_result_flag,model_name)

    define_seed() #seed固定
    data_x,data_t = mk_dataset(data_pickle_path,age_json_path,need_elements_list,minimum_signal_length,maximum_signal_length,out_path)
    trainloader,testloader = get_loader(data_x,data_t,train_rate,batch_size)
    num_axis = len(need_elements_list)
    net = select_model(model_name,num_axis,hidden_dim,num_layers,maximum_signal_length).to(device)
    logger.info(net)

    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    loss_fn = nn.MSELoss()
    epoch_loss = [] #グラフに出力するための損失格納用リスト
    try:
        for epoch in range(epochs):
            logger.info(f"----- epoch:{epoch+1} ---------------------------")
            train_running_loss = train_method(trainloader,net,optimizer,loss_fn,device,batch_size)
            test_running_loss,predicted_for_plot = test_method(testloader,net,loss_fn,device,print_result_flag)
            epoch_loss.append([train_running_loss,test_running_loss])
    except KeyboardInterrupt:
        pass

    if len(epoch_loss) != 0:
        plot_loss_glaph(epoch_loss,out_path) #1エポックでもあれば損失グラフ生成
        plot_inference_result(predicted_for_plot[:,1],predicted_for_plot[:,0],out_path) #最後のテスト結果をプロット
    else:
        logging.shutdown()
        shutil.rmtree(out_path) #1エポックもなければディレクトリごと削除


if __name__ == "__main__":
    main()