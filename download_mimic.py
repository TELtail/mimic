import pickle
from selenium import webdriver
import time
import urllib.request
import selenium
import os




def get_data(driver,element,names):
    element.click()
    file_elements = driver.find_elements_by_xpath("/html/body/pre/a")
    for file_element in file_elements:
        xpath = file_element.get_attribute("href")
        num = xpath.split("/")[-2]
        if num in names:
            num_dat = num + "n.dat"
            num_hea = num + "n.hea"
            file_element.click()
            try:
                link_dat = driver.find_element_by_link_text(num_dat)
                link_dat.click()
                link_hea = driver.find_element_by_link_text(num_hea).get_attribute("href")
                urllib.request.urlretrieve(link_hea,"./data/mimic-II/"+num_hea)
            except:
                pass
            driver.back()


def scraping(names):
    options = webdriver.ChromeOptions()
    options.add_experimental_option("prefs", {"download.default_directory":os.getcwd()+ "/data/mimic-II/" })

    driver = webdriver.Chrome(executable_path='./driver/chromedriver.exe',chrome_options=options)
    driver.get('https://archive.physionet.org/physiobank/database/mimic2wdb')

    num_list = ["30","31","32","33","34","35","36","37","38","39"] 
    elements = driver.find_elements_by_xpath("//*[@id='page']/pre[3]/a")

    for element in elements:
        xpath = element.get_attribute("href")
        if xpath.split("/")[-2] in num_list:
            get_data(driver,element,names)
            driver.back()

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