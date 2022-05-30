import pickle
from selenium import webdriver
import time




def get_data(driver,element,names):
    element.click()
    file_elements = driver.find_elements_by_xpath("/html/body/pre/a")
    for file_element in file_elements:

        if file_element.get_attribute("href").split("/")[-2] in names:
            pass




def scraping(names):
    driver = webdriver.Chrome(executable_path='./driver/chromedriver.exe')
    driver.get('https://archive.physionet.org/physiobank/database/mimic2wdb')

    num_list = ["30","31","32","33","34","35","36","37","38","39"] 
    elements = driver.find_elements_by_xpath("//*[@id='page']/pre[3]/a")

    for element in elements:
        xpath = element.get_attribute("href")
        if xpath.split("/")[-2] in num_list:
            get_data(driver,element,names)

def load_names(names_path):
    with open(names_path,"rb") as p:
        names = pickle.load(p)

    return names

def main():
    names_path = "./data/names.bin"
    names = load_names(names_path)
    scraping(names)

if __name__ == "__main__":
    main()