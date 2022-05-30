from IPython.display import display
import wfdb
import numpy as np


signals,fields = wfdb.rdsamp("./data/3009959n")
print(fields)
display(signals)
non_num = np.count_nonzero(np.isnan(signals))
element_num = np.count_nonzero(~np.isnan(signals))
print(non_num,element_num)

record = wfdb.rdrecord("./data/3009959n",channels=[0,1,2])
wfdb.plot_wfdb(record=record,title="Record")


"""
record = wfdb.rdrecord("p000020-2183-04-28-17-47",channels=[0,1,2],pn_dir="mimic3wdb/matched/p00/p000020/")
wfdb.plot_wfdb(record=record,title="Record")
"""