from IPython.display import display
import wfdb
import numpy as np
import glob

def plot_glaph(path,label_names):
    signals,fields = wfdb.rdsamp(path)

    label_index = []
    print(path)
    record = wfdb.rdrecord(path)#,channels=label_index)


path = "../data\mimic-II/*.dat"
label_names = ["HR","PULSE","SpO2"]

paths = glob.glob(path)
for p in paths:
    p = p.split(".dat")[0]
    plot_glaph(p,label_names)