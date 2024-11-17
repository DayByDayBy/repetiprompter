import json

def save_json(data, filename):
    """
    save a nested dictionary to a JSON file.

    args:
        data (dict): nested dictionary to save.
        filename (str): filename to save JSON data to.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def save_json_tree(data, filename):
    """
    save nested dictionary to JSON file, preserving tree structure.

    args:
        data (dict): nested dictionary to save.
        filename (str): filename to save JSON data to.
    """
    def _save_json_tree(data, indent=0):
        if isinstance(data, dict):
            json_str = '{\n'
            for key, value in data.items():
                json_str += '  ' * (indent + 1) + f'"{key}":\n'
                json_str += _save_json_tree(value, indent + 1)
            json_str += '  ' * indent + '}\n'
            return json_str
        elif isinstance(data, list):
            json_str = '[\n'
            for item in data:
                json_str += '  ' * (indent + 1) + _save_json_tree(item, indent + 1)
            json_str += '  ' * indent + ']\n'
            return json_str
        else:
            return str(data).replace('\n', '')
    with open(filename, 'w') as f:
        f.write(_save_json_tree(data))

if __name__ == '__main__':
    data = {'a': 1, 'b': {'c': 2, 'd': [3, 4]}}


    save_json(data, 'data.json')
    save_json_tree(data, 'data_tree.json')