import urllib.request


def get_needed_signal_names():
    age_path = "../data/MAP-CW.txt"
    with open(age_path,"r") as f:
        map_data = f.read()
        
        map_data = map_data.split("\n")

    target_datas = []

    for one in map_data:
        names = [one.split("\t")[-1]]
        if " " in names[0]:
            names = names[0].split(" ")
        for n in names:
            path = n[:2] + "/" + n
            target_datas.append(path)
    return target_datas

target_datas = get_needed_signal_names()
#print(target_datas)

save_dir = "C:/Users/s2020se12/Downloads/tmp"


for i,target in enumerate(target_datas):
    url = "https://archive.physionet.org/physiobank/database/mimic2wdb/" + target + "/RECORDS"
    print(url,target,"(",i,"/",len(target_datas),")")
    urllib.request.urlretrieve(url,save_dir+"/record_"+target.split("/")[-1]+".txt") #datファイルダウンロード
    