import ollama
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import os
from tqdm import tqdm


TIME_STAMP = datetime.now().strftime("%Y%m%d_%H%M")
MODEL_NAME = 'tinyllama'
BREADTH = 2
DEPTH = 2
SHAPE = f'{BREADTH} by {DEPTH}'
PROMPT_NICKNAME = 'recursion_prompt'
recursion_prompt =  "consider: the ability to recursively improve upon the present is the key to unlocking the boundless potential of the future, a tool of the gods, the engine of progress, the ultimate weapon in the battle against entropy."
INITIAL_PROMPT = recursion_prompt


# def generate_response(prompt: str) -> str:
#     return ollama.generate(model=MODEL_NAME, prompt=prompt)['response']

def generate_response(prompt: str) -> List[str]:
    responses = [prompt]
    for i in range(BREADTH):
        current_prompt = responses[-1]
        response = ollama.generate(model=MODEL_NAME, prompt=current_prompt)['response']
        responses.append(response) 
    return responses



def generate_layer(prompts: List[str], current_depth: int) -> Dict[str, List[Dict[str, str]]]:
    layer = {}
    for prompt in tqdm(prompts):
        key = f"{current_depth}.{prompts.index(prompt)+1}"
        responses = generate_response(prompt)
        chain = {"prompt": prompt, "responses": responses}
        if key not in layer:
            layer[key] = [chain]
        else:
            layer[key].append(chain)
    return layer


# def generate_layer(prompts: List[str], current_depth: int) -> List[Dict[str, List[str]]]:
#     layer = []
#     for i, prompt in enumerate(tqdm(prompts, desc = f"at depth of {current_depth}")):
#         responses = []
#         for j in tqdm(range(1, BREADTH + 1), desc= f"generating responses for prompt {i+1} ", leave= False):
#             key = f"{current_depth}.{i+1}.{j}"
#             try:
#                 response = generate_response(prompt)
#                 responses.append(response)
#             except Exception as e:
#                 print(f"error generating response for {key}: {e}")
#                 continue
#         layer.append({"prompt": prompt, "responses":responses})    
#     return layer


def generate_tree(current_prompt: str, depth: int) -> Dict[str, Any]:
    tree = {"prompt": current_prompt, "responses": []}
    current_layer = [tree]
    
    for k in range(1, depth + 1):
        next_layer = []     
        for node in current_layer:
            prompts = [node["prompt"]] * BREADTH    
            new_entries = generate_layer(prompts, k)
            node["responses"] = new_entries
            next_layer.extend(new_entries)
        current_layer = next_layer
    return tree


def save_tree(tree: Dict[str, Any], 
              prompt_nickname: str, 
              model_name: str,
              shape: str,
              timestamp: str, 
              filename: Optional[str] = None
              ):
    
    tree_key = f'{prompt_nickname}_{model_name}'
    full_tree ={
        "tree_key": tree_key,
        "timestamp": timestamp,
        "shape": shape,
        "content": tree,
        # "content":{"prompt": tree[0]["response"]}
    }
    
    if filename is None:
        filename = f'./responses/cubic_{MODEL_NAME}_at_{TIME_STAMP}.json'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(tree, f, indent=2)


if __name__ == '__main__':
    print('\n\n')
    print(f'running {MODEL_NAME} model')
    print('\n\n')
    tree = generate_tree(INITIAL_PROMPT, DEPTH)
    save_tree(tree, PROMPT_NICKNAME, MODEL_NAME, TIME_STAMP, SHAPE)
    print('\n\n')
    print("generated tree saved.")
    print('\n\n')

