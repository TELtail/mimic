import glob
import csv

"""
ダウンロードしてきたRECORDSデータを一つにまとめるプログラム
"""



dir_path = "../data/mimic_records/*.txt"
out_path = "../data/file_names.csv"
paths = glob.glob(dir_path)

data = []

for p in paths:
    with open(p,"r") as f:
        txt_data = f.read()
    for d in txt_data.split("\n"):
        if "_0" in d:
            data.append(d)


with open(out_path,"w", newline="") as f:
    writer = csv.writer(f)
    for d in data:
        writer.writerow([d])