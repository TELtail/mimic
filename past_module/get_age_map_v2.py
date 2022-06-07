import pickle
import json

def main_worker(age_path):
    with open(age_path,"r") as f:
        map_data = f.read()
    
    map_data = map_data.split("\n")

    age_data = {}
    count_90over = 0
    nan_count = 0
    for one_person in map_data:
        patient_number,_,age,signal_numbers = one_person.split("\t")
        if age[-1] == "+":
            count_90over += 1
            age = age[:-1] #90歳以上の人につく+を排除
        
        if age == "??":
            nan_count += 1
            age=0
        
        for num in signal_numbers.split(" "):
            age_data[num+"n"] = {"age":int(age),"patient_number":patient_number}

    ages = [i["age"] for i in age_data.values() if i["age"] !=0]
    ave_age = int(sum(ages)/len(ages))
    

    for key,value in age_data.items():
        if value["age"] == 0:
            value["age"] = ave_age
        age_data[key] = value
    
    age_data_plus_delete_info = {}
    age_data_plus_delete_info["info"] = {"age_average":ave_age,"nan_count":nan_count,"90_over_count":count_90over}
    age_data_plus_delete_info["data"] = age_data
    


    with open("age_data.json","w") as f:
        json.dump(age_data_plus_delete_info,f,indent=4)
    


def main():
    age_path = "../data/MAP-CW.txt"
    main_worker(age_path)

if __name__ == "__main__":
    main()