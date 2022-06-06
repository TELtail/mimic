from IPython.display import display
import wfdb
import numpy as np

path = "./data\mimic-II/3000105n"
label_names = ["HR","PULSE","SpO2"]


signals,fields = wfdb.rdsamp(path)
print(fields)
display(signals)


label_index = []
for name in label_names:
    label_index.append(fields["sig_name"].index(name))
record = wfdb.rdrecord(path,channels=label_index)
wfdb.plot_wfdb(record=record,title="Record")
