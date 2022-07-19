import os
import glob
import csv

dir_path = "/groups/gab50262/volunteerdata_2/volunteerdata/2019*/wfdb_data/*bed_data.bin"
out_path = "../data/paths_abci_bed.csv"

paths = glob.glob(dir_path)
print(paths)

with open(out_path,"w", newline="") as f:
    writer = csv.writer(f)
    for p in paths:
        writer.writerow([p])
