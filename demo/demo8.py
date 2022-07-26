import wfdb
import csv

csv_path = "../data/file_names.csv"

all_paths = []

with open(csv_path,"r") as f:
    reader = csv.reader(f)
    for name in reader:
        sig_name = name[0].split("_")[0]
        dat_name = sig_name[:2] +"/"+ sig_name +"/"+ name[0]
        all_paths.append(dat_name)
        print(dat_name)

#wfdb.dl_files("mimic3wdb","../data/tmp",all_paths[:2])
#wfdb.dl_database('mimic3wdb',"../data/tmp",records=all_paths[:1] )
#wfdb.dl_database('macecgdb',"../data/tmp" )
wfdb.dl_files('mimic3wdb',"../data/tmp",["30/3000003/3000003_0001.dat"])