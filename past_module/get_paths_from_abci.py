import os
import glob
import csv

dir_path = "../data/mimic_raw_mini/*/*/*.dat"
out_path = "../data/paths_abci.csv"

paths = glob.glob(dir_path)
print(paths)

with open(out_path,"w", newline="") as f:
    writer = csv.writer(f)
    for p in paths:
        writer.writerow([p])
