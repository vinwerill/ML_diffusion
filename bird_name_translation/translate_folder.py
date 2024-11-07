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
            english_name = translations.get(folder_name)
            if english_name:
                new_folder_path = os.path.join(base_folder, english_name)
                os.rename(folder_path, new_folder_path)
                print(f'Renamed: {folder_path} -> {new_folder_path}')

if __name__ == "__main__":
    base_folder = './dataset'
    json_file = './bird_name_translation/chinese_to_english.json'
    
    translations = load_translations(json_file)
    translate_folder_names(base_folder, translations)