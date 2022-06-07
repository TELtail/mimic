from IPython.display import display
import wfdb
import numpy as np

def plot_glaph(path,label_names):
    signals,fields = wfdb.rdsamp(path)

    label_index = []
    print("-------------")
    print(path)
    for name in label_names:
        idx = fields["sig_name"].index(name)
        label_index.append(idx)
        print(name,signals[:,idx].shape)
    record = wfdb.rdrecord(path,channels=label_index)
    print("-------------")
    wfdb.plot_wfdb(record=record,title="Record")


path = "./data\mimic-II/3000105n"
label_names = ["HR","PULSE","SpO2"]

plot_glaph(path,label_names)