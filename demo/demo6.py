import wfdb
import os
import selenium
from selenium import webdriver

record = wfdb.rdrecord("p000020-2183-04-28-17-47",channels=[0,1,2],pn_dir="mimic3wdb/matched/p00/p000020/")
wfdb.plot_wfdb(record=record,title="Record")

record = wfdb.rdrecord("3000003_0001",pn_dir="mimic3wdb/30/3000003/")
wfdb.plot_wfdb(record=record,title="Record")
