import pickle

def main_worker(age_path):
    with open(age_path,"r") as f:
        map_data = f.read()
    
    map_data = map_data.split("\n")

    age_data = {}

    for one_person in map_data:
        numbers = one_person.split("\t")[-1] #患者番号を取得
        age = one_person.split("\t")[-2] #年齢取得
        if age[-1] == "+":
            age = age[:-1] #90歳以上の人につく+を排除
        
        if age == "??":
            age=0
        
        for num in numbers.split(" "):
            age_data[num+"n"] = int(age)

    ages = [i for i in age_data.values() if i!=0]
    ave_age = int(sum(ages)/len(ages))
    
    new_age_data = {}

    for key,value in age_data.items():
        if value == 0:
            value = ave_age
        new_age_data[key] = value
    
    with open("age_data.bin","wb") as f:
        pickle.dump(new_age_data,f)
    


def main():
    age_path = "./data/MAP-CW.txt"
    main_worker(age_path)

if __name__ == "__main__":
    main()