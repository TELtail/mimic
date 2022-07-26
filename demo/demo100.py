import wfdb
import csv


csv_path = "../data/file_names.csv"

all_paths = []

with open(csv_path,"r") as f:
    reader = csv.reader(f)
    for name in reader:
        sig_name,sub_num = name[0].split("_")
        if int(sub_num) > 5:
            continue
        dat_name = sig_name[:2] +"/"+ sig_name +"/"+ name[0] + ".dat"
        hea_name = sig_name[:2] +"/"+ sig_name +"/"+ name[0] + ".hea"
        all_paths.append(dat_name)
        all_paths.append(hea_name)

wfdb.dl_files("mimic3wdb","../data/mimic_raw_mini",all_paths)




#wfdb.dl_files('mimic3wdb',"../data/tmp",["30/3000003/3000003_0001.dat"])

