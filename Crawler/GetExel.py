import requests
import os
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup


options = Options()
options.chrome_executable_path="./Crawler/chromedriver.exe"

prefs = {"download.default_directory": "C:\\VSCode_WorkSpace_2\\Python\\BirdSoundRecognition\\data\\excel"}
options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(options=options) 


myAccount = ""
myPassword = ""
TotalBirdListUrl = "https://ebird.org/region/TW/bird-list"

def rename_file(after_name):
    os.rename("./data/excel/export.csv", f"./data/excel/{after_name}.csv")
    pass

def sign_in():    
    url = "https://secure.birds.cornell.edu/cassso/login"
    driver.get(url)
    
    accoutInputBlock = driver.find_element(By.ID, "input-user-name")
    accoutInputBlock.send_keys(myAccount)
    
    passwordInputBlock = driver.find_element(By.ID, "input-password")
    passwordInputBlock.send_keys(myPassword)

    submitInputBlock = driver.find_element(By.ID, "form-submit")
    submitInputBlock.send_keys(Keys.ENTER)
    
    time.sleep(1)
    pass

def get_all_short_name():
    retarr = []

    url = TotalBirdListUrl
    headers = {}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("failed to get all short name")
        return retarr
    
    soup = BeautifulSoup(response.text, "html.parser")

    # find native bird
    bird_list = soup.find("section", attrs={"aria-labelledby": "nativeNaturalized"})
    bird_list = bird_list.find_all("li")    
    for bird in bird_list:
        retarr.append(bird.get("id"))
    
    # find provisional bird
    bird_list = soup.find("section", attrs={"aria-labelledby": "provisional"})
    bird_list = bird_list.find_all("li")    
    for bird in bird_list:
        retarr.append(bird.get("id"))

    print(f"{len(retarr)} birds in total")
    return retarr

def get_audio_site(short_name):
    return f"https://media.ebird.org/catalog?taxonCode={short_name}&mediaType=audio&view=list"

def get_download_site(short_name):
    return f"https://media.ebird.org/api/v2/export.csv?taxonCode={short_name}&mediaType=audio&birdOnly=true&count=10000"

def get_excel(audio_site, short_name):
    
    if os.path.exists("./data/excel/export.csv"):
        print(f"{short_name} can't be download correctly")
        return False
    
    driver.get(url=audio_site)
    download_site = get_download_site(short_name)
    driver.get(download_site) # download the file
    
    while not os.path.exists("./data/excel/export.csv"):
        time.sleep(0.1)

    time.sleep(0.5)
    
    while not os.path.exists(f"./data/excel/{short_name}.csv"):
        rename_file(f"{short_name}")
    return True

def main():
    sign_in()
    
    bird_short_name_list = get_all_short_name()
    failed_species = []
    
    time.sleep(1)
    
    for bird_name in bird_short_name_list:
        print(f"downloading excel of {bird_name}")
        try:
            if not get_excel(get_audio_site(bird_name), bird_name):
                print("download failed, terminate the process")
                return
            time.sleep(0.5)
            print("download successful")
        except:
            failed_species.append(bird_name)
            print(f"failed to download excel of {bird_name}")
        print("-" * 30)
            
    print(f"number of failed speceis {len(failed_species)}")
    print(failed_species)


if __name__ == "__main__":
    main()
