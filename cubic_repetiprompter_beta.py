import ollama
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import os
from tqdm import tqdm
import logging

import time
start_time = time.time()

os.environ['OLLAMA_NUM_PARALLEL'] = '2'

logging.basicConfig(filename='tree_generation.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

TIME_STAMP = datetime.now().strftime("%Y%m%d_%H%M")
MODEL_NAME = 'llama3.1'
TEMP = 0.67
CHAIN_LENGTH = 2
RECURSION_DEPTH = 2
SHAPE = f'{CHAIN_LENGTH} by {RECURSION_DEPTH}'
PROMPT_NICKNAME = 'recursion_prompt'
INITIAL_PROMPT = "consider: the ability to recursively improve upon the present is the key to unlocking the boundless potential of the future, a tool of the gods, the engine of progress, the ultimate weapon in the battle against entropy."

def generate_response(prompt: str) -> str:
    try:
        return ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": TEMP})['response']
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return ""

def generate_chain(seed_prompt: str, chain_length: int) -> List[str]:
    chain = [seed_prompt]
    for _ in tqdm(range(chain_length), desc="generating chain", leave=False):
        response = generate_response(chain[-1])
        if response:
            chain.append(response)
        else:
            break
    return chain

def generate_tree(seed_prompt: str, chain_length: int, recursion_depth: int) -> Dict[str, Any]:
    chain = generate_chain(seed_prompt, chain_length)
    tree = {"prompt": seed_prompt, "responses": chain[1:]}
    
    if recursion_depth > 1:
        tree["children"] = []
        for response in tqdm(chain[1:], desc=f"recursion depth {recursion_depth}", leave=False):
            child_tree = generate_tree(response, chain_length, recursion_depth - 1)
            tree["children"].append(child_tree)
    
    return tree

def save_tree(tree: Dict[str, Any], metadata: Dict[str, Any], filename: Optional[str] = None):
    full_tree = {
        "metadata": metadata,
        "content": tree
    }
    
    if filename is None:
        filename = f'./responses/tree_{metadata["model_name"]}_at_{metadata["timestamp"]}.json'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(full_tree, f, indent=2)

if __name__ == '__main__':
    print(f'\n\nRunning {MODEL_NAME} model\n\n')
    
    metadata = {
        "tree_key": f'{PROMPT_NICKNAME}_{MODEL_NAME}',
        "timestamp": TIME_STAMP,
        "shape": SHAPE,
        "model_name": MODEL_NAME,
        "chain_length": CHAIN_LENGTH,
        "recursion_depth": RECURSION_DEPTH
    }
    
    tree = generate_tree(INITIAL_PROMPT, CHAIN_LENGTH, RECURSION_DEPTH)
    save_tree(tree, metadata)
    
    print("\n\ngenerated tree saved.\n\n")
    
    end_time = time.time()
    print(f"executed in {end_time - start_time} seconds\n")