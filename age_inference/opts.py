import argparse
from common_utils import log_start



def get_parser():
    parser = argparse.ArgumentParser("MIMIC-IIデータセットで年齢の学習、推論を行うプログラム")
    parser.add_argument("-d","--data_pickle_path",help="信号のバイナリデータのパス")
    parser.add_argument("-a","--age_json_path",help="年齢のjsonデータのパス")
    parser.add_argument("--out_path",help="グラフ等を出力するパス",default="../out")
    parser.add_argument("--train_rate",help="学習データの割合",type=float,default=0.8)
    parser.add_argument("--batch_size",help="バッチサイズ",type=int,default=8)
    parser.add_argument("--hidden_dim",help="LSTMの次元",type=int,default=64)
    parser.add_argument("--num_layers",help="LSTMの層数",type=int,default=3)
    parser.add_argument("--epochs",help="epoch数",type=int,default=100)
    parser.add_argument("--lr",help="学習率",type=float,default=1e-3)
    parser.add_argument("--minimum_signal_length",help="最小の信号の長さ",type=int,default=300)
    parser.add_argument("--maximum_signal_length",help="最大の信号の長さ",type=int,default=1500)
    parser.add_argument("--need_elements_list",help="必要な要素名",nargs="*",default=['HR', 'RESP', 'SpO2'])
    parser.add_argument("--config_path",help="log_config.jsonのpath指定",default="./log_config.json")
    parser.add_argument("--print_result_flag",help="test結果をprintするかどうか",action='store_true')
    parser.add_argument("--model_name",help="使用するモデル名を指定",default="Lstm_regression_net",choices=["Lstm_regression_net","Lstm_classification_net","Conv1D_regression_net","Conv1D_classification_net","Linear_regression_net","Linear_classification_net"])
    parser.add_argument("--debug_flag",help="デバッグ中を示す。結果の保存を行わない",action="store_true")
    parser.add_argument("--splited_one_signal_length",help="分割処理をする際にどれくらいの長さにするか",default=None)

    args = parser.parse_args()


    return args

def print_parser(args):
    logger = log_start()
    logger.info("---------------------------------")
    logger.info("data_pickle_path:{}".format(args.data_pickle_path))
    logger.info("age_json_path:{}".format(args.age_json_path))
    logger.info("config_path:{}".format(args.config_path))
    logger.info("out_path:{}".format(args.out_path))
    logger.info("train_rate:{}".format(args.train_rate))
    logger.info("batch_size:{}".format(args.batch_size))
    logger.info("hidden_dim:{}".format(args.hidden_dim))
    logger.info("epochs:{}".format(args.epochs))
    logger.info("lr:{}".format(args.lr))
    logger.info("minimum_signal_length:{}".format(args.minimum_signal_length))
    logger.info("maximum_signal_length:{}".format(args.maximum_signal_length))
    logger.info("need_elements_list:{}".format(args.need_elements_list))
    logger.info("print_result_flag:{}".format(args.print_result_flag))
    logger.info("model_name:{}".format(args.model_name))
    logger.info("splited_one_signal_length:{}".format(args.splited_one_signal_length))
    logger.info("---------------------------------")