import os
import glob
import csv

dir_path = "../data/mimic-exp/*.dat"
out_path = "../data/paths.csv"

paths = glob.glob(dir_path)

with open(out_path,"w", newline="") as f:
    writer = csv.writer(f)
    for p in paths:
        writer.writerow([p])