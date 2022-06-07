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

def sort_name(names,delete_num):
    names = [int(i) for i in names ]
    names = sorted(names)
    names = [str(i) for i in names if delete_num < i]
    return names

def main():
    names_path = "./data/MAP-CW.txt"
    delete_num = 3100000

    names = get_need_name(names_path)
    names = sort_name(names,delete_num)
    print(names)
    with open("names_mini.bin","wb") as p:
        pickle.dump(names,p)


if __name__ == "__main__":
    main()