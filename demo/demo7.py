import wfdb
import csv

csv_path = "../data/file_names.csv"

with open(csv_path,"r") as f:
    reader = csv.reader(f)
    for name in reader:
        sig_name = name[0].split("_")[0]
        pn_dir = "mimic3wdb/" + sig_name[:2] +"/"+ sig_name +"/"
        record = wfdb.rdrecord(name[0],pn_dir=pn_dir)
        wfdb.plot_wfdb(record=record,title="Record")
