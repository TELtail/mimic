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
    out_path = mk_out_dir(args.out_path)
    global logger
    set_log_settings(out_path,args.config_path)
    logger = log_start()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device:{}".format(device))
    
    

    print_parser(args)

    define_seed() #seed固定
    data_x,data_t = mk_dataset_v2(args.data_pickle_path,args.age_json_path,args.need_elements_list,args.minimum_signal_length,args.maximum_signal_length,out_path,args.model_type)
    trainloader,testloader = get_loader(data_x,data_t,args.train_rate,args.batch_size)
    out_dim,loss_fn = determing_setting(args.model_type)
    num_axis = len(args.need_elements_list)
    net = select_model(args.model_name,num_axis,args.hidden_dim,args.num_layers,args.maximum_signal_length,args.model_type,out_dim).to(device)
    logger.info(net)

    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
    epoch_loss = [] #グラフに出力するための損失格納用リスト
    correct_for_classification = []
    try:
        for epoch in range(args.epochs):
            logger.info(f"----- epoch:{epoch+1} ---------------------------")
            if args.model_type == "regression":
                train_running_loss = train_regression_method(trainloader,net,optimizer,loss_fn,device,args.batch_size)
                test_running_loss,predicted_for_plot = test_regression_method(testloader,net,loss_fn,device,args.print_result_flag)
            elif args.model_type == "classification":
                train_running_loss,train_correct = train_classification_method(trainloader,net,optimizer,loss_fn,device,args.batch_size)
                test_running_loss,test_correct = test_classification_method(testloader,net,loss_fn,device,args.print_result_flag)
                correct_for_classification.append([train_correct,test_correct])
            epoch_loss.append([train_running_loss,test_running_loss])
    except KeyboardInterrupt:
        pass

    if len(epoch_loss) != 0 and args.debug_flag != True:
        plot_loss_glaph(epoch_loss,out_path) #1エポックでもあれば損失グラフ生成
        if args.model_type == "regression":
            plot_regression_inference_result(predicted_for_plot[:,1],predicted_for_plot[:,0],out_path) #最後のテスト結果をプロット
        elif args.model_type == "classification":
            correct_for_classification = np.array(correct_for_classification)
            plot_classification_correct_result(correct_for_classification[:,0],correct_for_classification[:,1],out_path)
    else:
        logging.shutdown()
        shutil.rmtree(out_path) #1エポックもなければディレクトリごと削除


if __name__ == "__main__":
    main_method()