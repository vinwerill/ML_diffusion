import requests
import os
import time
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

MAX_AUDIO_CNT = 100
NAME_DICT = {}

# dataframe = pd.read_csv(filepath_or_buffer="./data/excel/aleter1.csv")
# dataframe = dataframe.sort_values(by=["Average Community Rating"], ascending=False)
# dataframe = dataframe.head(10)

# print(dataframe)

def get_chinese_name_dict():
    url = "https://ebird.org/region/TW/bird-list"
    headers = {"Cookie": "I18N_LANGUAGE=zh"}
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print("failed to get dictionary")
        return
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    bird_list = soup.find("section", attrs={"aria-labelledby": "nativeNaturalized"})
    bird_list = bird_list.find_all("li")
    for bird in bird_list:
        chinese_name = bird.find("span", class_="Species-common").text
        short_name = bird.get("id")
        NAME_DICT[short_name] = chinese_name

    bird_list = soup.find("section", attrs={"aria-labelledby": "provisional"})
    bird_list = bird_list.find_all("li")
    for bird in bird_list:
        chinese_name = bird.find("span", class_="Species-common").text
        short_name = bird.get("id")
        NAME_DICT[short_name] = chinese_name

def get_dataframe(filepath):
    dataframe = pd.read_csv(filepath_or_buffer=filepath)
    dataframe = dataframe.sort_values(by=["Average Community Rating"], ascending=False)
    dataframe = dataframe.head(MAX_AUDIO_CNT)
    return dataframe

def download_audio(dataframe, dir_name, short_name):
    audio_cnt = dataframe.shape[0]
    print(f"total {audio_cnt} files to download")
    
    id_list = dataframe.get("目錄編號").tolist()

    for i in tqdm(range(audio_cnt)):
        current_id = id_list[i]
        filename = f"{dir_name}/{short_name}_{i}.mp3"
        
        url = f"https://cdn.download.ams.birds.cornell.edu/api/v2/asset/{current_id}/mp3"
        
        response = requests.get(url)
        if response.status_code != 200:
            print(f"failed to download one file of {short_name}")
            continue
        
        with open(file=filename, mode="wb") as file:
            file.write(response.content)
        
        continue
    pass



def main():
    get_chinese_name_dict()
    
    csv_directory = "./data/excel"
    audio_directory = "E:\\Sean\\BirdRecognitionTrainingData\\Audio"
    filecnt = 0
    for filename in os.scandir(csv_directory):
        if not filename.is_file():
            continue
        
        filecnt += 1
        filepath = filename.path
        fullname = filename.name
        short_name = fullname.split(".")[0]
        chinese_name = NAME_DICT[short_name]

        if not os.path.exists(f"{audio_directory}/{chinese_name}"):
            os.makedirs(f"{audio_directory}/{chinese_name}")
        
        print(f"downloading audio of {chinese_name}, {filecnt}/672")
        download_audio(dataframe=get_dataframe(filepath), dir_name=f"{audio_directory}/{chinese_name}", short_name=short_name)
        print("-" * 30)



if __name__ == "__main__":
    main()
