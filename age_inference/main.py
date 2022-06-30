import torch
import torch.nn as nn
import numpy as np
import shutil
from logging import getLogger,config
import logging
from common_utils import mk_out_dir,select_model,set_log_settings,define_seed,log_start,determing_setting
from opts import print_parser,get_parser
from datasets import mk_dataset,get_loader,mk_dataset_v2
from plot_glaph import plot_loss_glaph,plot_regression_inference_result,plot_classification_correct_result
from train_test_loop_method import test_regression_method,train_regression_method,train_classification_method,test_classification_method





def main_method():
    args = get_parser()
    out_path = mk_out_dir(args.out_path) #出力ディレクトリ作成
    global logger
    set_log_settings(out_path,args.config_path) #loggerの設定をjsonファイルから取得
    logger = log_start()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device:{}".format(device))

    model_type = args.model_name.split("_")[1]


    print_parser(args) #取得したコマンドライン引数を表示

    define_seed() #seed固定
    data_x,data_t,train_indices,test_indices = mk_dataset_v2(args.data_pickle_path,args.age_json_path,args.need_elements_list,
                                                            args.minimum_signal_length,args.maximum_signal_length,out_path,
                                                            model_type,args.train_rate,args.splited_one_signal_length)
    trainloader,testloader = get_loader(data_x,data_t,args.train_rate,args.batch_size,train_indices,test_indices)
    logger.info("train:{} test:{}".format(len(trainloader.dataset),len(testloader.dataset)))
    out_dim,loss_fn = determing_setting(model_type) #モデルタイプに対応したモデルの出力次元と損失関数を決定 
    loss_fn = loss_fn.to(device)
    num_axis = len(args.need_elements_list) #入力次元数を決定
    net = select_model(args.model_name,num_axis,args.hidden_dim,args.num_layers,args.maximum_signal_length,out_dim).to(device) #指定されたモデルを呼び出し
    logger.info(net) #モデル情報出力

    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
    epoch_loss = [] #グラフに出力するための損失格納用リスト
    correct_for_classification = [] #分類問題時の正答率を格納するためのリスト
    try:
        for epoch in range(args.epochs):
            logger.info(f"----- epoch:{epoch+1} ---------------------------")
            if model_type == "regression": #回帰
                train_running_loss = train_regression_method(trainloader,net,optimizer,loss_fn,device,args.batch_size) #学習
                test_running_loss,predicted_for_plot = test_regression_method(testloader,net,loss_fn,device,args.print_result_flag) #テスト
            elif model_type == "classification": #分類
                train_running_loss,train_correct = train_classification_method(trainloader,net,optimizer,loss_fn,device,args.batch_size) #学習
                test_running_loss,test_correct = test_classification_method(testloader,net,loss_fn,device,args.print_result_flag) #テスト
                correct_for_classification.append([train_correct,test_correct]) #分類問題の結果を格納
            epoch_loss.append([train_running_loss,test_running_loss]) #損失を格納
    except KeyboardInterrupt:
        pass

    if len(epoch_loss) != 0 and args.debug_flag != True:
        plot_loss_glaph(epoch_loss,out_path) #1エポックでもあれば損失グラフ生成
        if model_type == "regression":
            plot_regression_inference_result(predicted_for_plot[:,1],predicted_for_plot[:,0],out_path) #最後のテスト結果をプロット
        elif model_type == "classification":
            correct_for_classification = np.array(correct_for_classification)
            plot_classification_correct_result(correct_for_classification[:,0],correct_for_classification[:,1],out_path) #分類時の正答率の変遷グラフを作成
    else:
        logging.shutdown()
        shutil.rmtree(out_path) #1エポックもなければディレクトリごと削除


if __name__ == "__main__":
    main_method()