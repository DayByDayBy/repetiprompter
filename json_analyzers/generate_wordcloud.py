import os
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud as WordCloud
  
# import networkx as nx

def read_json_from_dir(directory_path):
    data_list = []
    file_counter = 0 
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.json'):
            file_counter += 1
            file_path = os.path.join(directory_path, file_name)     
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    data_list.append(data)
                except json.JSONDecodeError:
                    print(f'error reading {file_name}')
            print(file_counter)
    return data_list, file_counter

def collect_text(subtree):
    all_text = ""
    if isinstance(subtree, dict):
        for key, value in subtree.items():
             if key.lower() != 'prompt':
                all_text += collect_text(value)
    elif isinstance(subtree, list):
        for item in subtree:
            all_text += collect_text(item)
    elif isinstance(subtree, str):
        all_text += " "  + subtree
    return all_text    

def generate_wordcloud(json_data_list):
    
    all_text = ""
    for json_data in json_data_list:
        all_text += collect_text(json_data)
        if all_text.strip():
            wordcloud = WordCloud(width = 800, 
                          height = 600, 
                          background_color = 'white').generate(all_text)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('word cloud of all responses')
            plt.show()
        else:
            print("no valid text found - check path, check dir, etc")
    
directory_path = 'wordcloud_source'
json_data_list = read_json_from_dir(directory_path)
generate_wordcloud(json_data_list)