import wfdb
import os

f = open("./data/RECORDS-waveforms.txt","r")
file_path_list = f.read()
f.close()
file_path_list = file_path_list.split("\n")
print(len(file_path_list))
data = []

try:
    for file_path in file_path_list:
        file_name = os.path.basename(file_path)
        file_parent = os.path.dirname(file_path)
        file_parent = "mimic3wdb/matched/"+file_parent
        signals,fields = wfdb.rdsamp(file_name,pn_dir=file_parent)
        data.append([signals,fields])
        print(file_path,signals.shape)
        print(fields)
except KeyboardInterrupt:
    pass

print(len(data))