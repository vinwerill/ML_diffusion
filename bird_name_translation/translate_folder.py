import os
import json

def load_translations(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        translations = json.load(file)
    return translations

def translate_folder_names(base_folder, translations):
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            base_name = folder_name.replace('_5secs', '')
            english_name = translations.get(base_name)
            if english_name:
                new_folder_name = english_name 
                new_folder_path = os.path.join(base_folder, new_folder_name)
                os.rename(folder_path, new_folder_path)
                print(f'Renamed: {folder_path} -> {new_folder_path}')

if __name__ == "__main__":
    base_folder = '../audio-diffusion-pytorch-trainer-main/classified_data/'
    json_file = 'chinese_to_english.json'
    
    translations = load_translations(json_file)
    translate_folder_names(base_folder, translations)