
import  age_inference as age

data_pickle_path = "./data/data.bin"
age_json_path = "./data/age_data.json"
train_rate = 0.8
batch_size = 12
need_elements_list = ["HR","RESP","SqO2"]
minimum_signal_length = 300
maximum_signal_length = 1500
out_path = "./out/"

out_path = age.mk_out_dir(out_path)

age.mk_dataset(data_pickle_path,age_json_path,train_rate,batch_size,
            need_elements_list,minimum_signal_length,maximum_signal_length,
            out_path)