import pickle

def get_need_name(names_path):
    names = []

    f = open(names_path,"r")
    data = f.read()
    data = data.split("\n") #改行ごとに分割
    for element in data:
        name = element.split("\t")[-1] #タブがある所=取得したい番号があるところ
        if " " in name:
            name_minis = name.split(" ") #番号が複数ある文字列の処理
            for name_mini in name_minis:
                names.append(name_mini)
            continue
        names.append(name)
    return names

def main():
    names_path = "./data/MAP-CW.txt"
    names = get_need_name(names_path)
    with open("names.bin","wb") as p:
        pickle.dump(names,p)


if __name__ == "__main__":
    main()