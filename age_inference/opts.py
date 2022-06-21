import argparse
from common_utils import log_start



def get_parser():
    parser = argparse.ArgumentParser("MIMIC-IIデータセットで年齢の学習、推論を行うプログラム")
    parser.add_argument("--data_path",help="信号のバイナリデータのパス")
    parser.add_argument("--age_path",help="年齢のバイナリデータのパス")
    parser.add_argument("--out_path",help="グラフ等を出力するパス",default="../out")
    parser.add_argument("--train_rate",help="学習データの割合",type=float,default=0.8)
    parser.add_argument("--batch_size",help="バッチサイズ",type=int,default=8)
    parser.add_argument("--hidden_dim",help="LSTMの次元",type=int,default=64)
    parser.add_argument("--num_layers",help="LSTMの層数",type=int,default=3)
    parser.add_argument("--epochs",help="epoch数",type=int,default=100)
    parser.add_argument("--lr",help="学習率",type=float,default=1e-3)
    parser.add_argument("--min",help="最小の信号の長さ",type=int,default=300)
    parser.add_argument("--max",help="最大の信号の長さ",type=int,default=1500)
    parser.add_argument("--need_elements",help="必要な要素名",nargs="*",default=['HR', 'RESP', 'SpO2'])
    parser.add_argument("--config",help="log_config.jsonのpath指定",default="./log_config.json")
    parser.add_argument("--print_result",help="test結果をprintするかどうか",action='store_true')
    parser.add_argument("--model_name",help="使用するモデル名を指定",default="Lstm_net")

    args = parser.parse_args()

    data_pickle_path = args.data_path
    age_json_path = args.age_path
    out_path = args.out_path
    train_rate = args.train_rate
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    epochs = args.epochs
    lr = args.lr
    minimum_signal_length = args.min
    maximum_signal_length = args.max
    need_elements_list = args.need_elements
    config_path = args.config
    print_result_flag = args.print_result
    model_name = args.model_name


    return (data_pickle_path,age_json_path,out_path,train_rate,batch_size,
            hidden_dim,num_layers,epochs,lr,minimum_signal_length,maximum_signal_length,
            need_elements_list,config_path,print_result_flag,model_name)

def print_parser(data_pickle_path,age_json_path,out_path,train_rate,
                batch_size,hidden_dim,num_layers,epochs,lr,minimum_signal_length,
                maximum_signal_length,need_elements_list,config_path,print_result_flag,
                model_name):
    logger = log_start()
    logger.info("---------------------------------")
    logger.info("data_pickle_path:{}".format(data_pickle_path))
    logger.info("age_json_path:{}".format(age_json_path))
    logger.info("config_path:{}".format(config_path))
    logger.info("out_path:{}".format(out_path))
    logger.info("train_rate:{}".format(train_rate))
    logger.info("batch_size:{}".format(batch_size))
    logger.info("hidden_dim:{}".format(hidden_dim))
    logger.info("epochs:{}".format(epochs))
    logger.info("lr:{}".format(lr))
    logger.info("minimum_signal_length:{}".format(minimum_signal_length))
    logger.info("maximum_signal_length:{}".format(maximum_signal_length))
    logger.info("need_elements_list:{}".format(need_elements_list))
    logger.info("print_result_flag:{}".format(print_result_flag))
    logger.info("model_name:{}".format(model_name))
    logger.info("---------------------------------")