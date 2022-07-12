import pickle
from selenium import webdriver
import time
import urllib.request
import selenium
import os




def get_data(driver,element,names,save_dir):
    element.click()
    file_elements = driver.find_elements_by_xpath("/html/body/pre/a")
    for file_element in file_elements:
        xpath = file_element.get_attribute("href")
        num = xpath.split("/")[-2]
        print(num)
        if num in names: #患者番号に該当したら
            print("---")
            num_dat = num + "_0001.dat"
            num_hea = num + "_0001.hea"
            file_element.click()
            print("--------------")
            try:
                link_dat = driver.find_element_by_link_text(num_dat).get_attribute("href")
                urllib.request.urlretrieve(link_dat,save_dir+"/"+num_dat) #heaファイルダウンロード
                link_hea = driver.find_element_by_link_text(num_hea).get_attribute("href")
                urllib.request.urlretrieve(link_hea,save_dir+"/"+num_hea) #heaファイルダウンロード
            except:
                pass
            driver.back() #前の画面に戻る


def scraping(names,save_dir):
    
    options = webdriver.ChromeOptions()
    options.add_experimental_option("prefs", {"download.default_directory":os.getcwd()+ save_dir }) #ダウンロード先を変更
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    options.use_chromium = True

    driver = webdriver.Chrome(executable_path='../driver/chromedriver.exe',chrome_options=options)
    driver.get('https://archive.physionet.org/physiobank/database/mimic2wdb') #MIMIC-IIのサイトを指定

    num_list = ["30","31","32","33","34","35","36","37","38","39"] #必要なディレクトリ群の番号
    elements = driver.find_elements_by_xpath("//*[@id='page']/pre[3]/a") 

    for element in elements:
        xpath = element.get_attribute("href")
        if xpath.split("/")[-2] in num_list: #必要なディレクトリ群のXPATHを取得したら
            get_data(driver,element,names,save_dir) #データダウンロード
            driver.back() 

def load_names(names_path):
    with open(names_path,"rb") as p:
        names = pickle.load(p)

    return names

def main():
    names_path = "../data/names.bin" #必要な患者番号が書かれたファイルのpath
    save_dir = "C:/Users/s2020se12/Downloads/tmp"
    names = load_names(names_path) 
    scraping(names,save_dir)

if __name__ == "__main__":
    main()