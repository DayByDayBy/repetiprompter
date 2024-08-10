import ollama
from typing import Dict
import time
import json
import csv
import os
from ratelimit import limits, sleep_and_retry
from datetime import datetime

TIME_STAMP = datetime.now().strftime("%Y%m%d_%H%M")

MODEL_NAME = 'llama3.1:8b'

DEPTH = 16

INITIAL_PROMPT = "i think the ability to recursively improve upon the present is the key to unlocking the boundless potential of the future, a tool of the gods, the engine of progress, the ultimate weapon in the battle against entropy. you?"


# @sleep_and_retry
# @limits(calls=1, period=0.5) 
def generate_response(prompt: str) -> str:
    print('generating...')
    return ollama.generate(model=MODEL_NAME, prompt=prompt)['response']

def generate_chain(start_key: str, start_prompt: str, depth: int) -> Dict[str, str]:
    chain = {start_key: start_prompt}
    current_key = start_key
    for i in range(1, depth + 1):
        try:
            new_key = f"{start_key}.{i}"
            chain[new_key] = generate_response(chain[current_key])
            current_key = new_key
        except Exception as e:
            print(f"error generating response for {new_key}: {e}")
            break
    return chain

def generate_tree(depth: int = DEPTH) -> Dict[str, str]:
    tree = {"1": INITIAL_PROMPT}
    tree.update(generate_chain("1", INITIAL_PROMPT, depth))
    
    for i in range(1, depth + 1):
        start_key = f"{i+1}"
        start_prompt = tree[f"1.{i}"]
        tree[start_key] = start_prompt
        tree.update(generate_chain(start_key, start_prompt, depth))
    return tree

def save_tree(tree: Dict[str, str], filename: str = None):
    if filename is None:
        filename = f'./responses/{MODEL_NAME}_at_{TIME_STAMP}.json'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(tree, f, indent=2)

def load_tree(filename: str = 'response_tree.json') -> Dict[str, str]:
    with open(filename, 'r') as f:
        return json.load(f)

def visualize_tree(tree: Dict[str, str]):
    for key, value in sorted(tree.items()):
        indent = '  ' * (len(key.split('.')) - 1)
        print(f"{indent}{key}: {value[:50]}...")

def save_tree_as_csv(tree: Dict[str, str], filename: str = None):
    if filename is None:
        filename = f'./responses/{MODEL_NAME}_at_{TIME_STAMP}.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Key', 'Prompt/Response'])
        for key, value in sorted(tree.items()):
            writer.writerow([key, value])
    print(f"Tree saved as CSV to {filename}")

if __name__ == '__main__':
    print('running...')
    tree = generate_tree(DEPTH)
    print('running...')
    save_tree(tree)
    print('running...')
    save_tree_as_csv(tree)
    print('running...')
    print("Generated tree:")
    visualize_tree(tree)