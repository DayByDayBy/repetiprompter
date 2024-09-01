import os
import json
import matplotlib.pyplot as plt
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


def generate_wordcloud(json_data_list):
    from wordcloud import WordCloud as WordCloud
    all_text = ""
    def collect_text(subtree):
        nonlocal all_text
        if isinstance(subtree, dict):
            if 'prompt' in subtree:
                all_text += " " + subtree['prompt']
            if 'responses' in subtree:
                all_text += " " + subtree['responses']
            if 'children' in subtree:
                for child in subtree['children']:
                    collect_text(child)
        elif isinstance(subtree, list):
            for item in subtree:
                collect_text(item)
        
    for json_data in json_data_list:
        collect_text(json_data)
    wordcloud = WordCloud(width = 800, height = 600, background_color = 'white').generate(all_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('word cloud of all responses')
    plt.show()
    
directory_path = '../responses'
json_data_list = read_json_from_dir(directory_path)
generate_wordcloud(json_data_list)