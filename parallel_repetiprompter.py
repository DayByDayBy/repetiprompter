import concurrent.futures
from functools import partial
import ollama
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import os
from tqdm import tqdm
import logging

TIME_STAMP = datetime.now().strftime("%Y%m%d_%H%M")
MODEL_NAME = 'llama3'
TEMP=0.6
CHAIN_LENGTH = 5
RECURSION_DEPTH = 5
SHAPE = f'{CHAIN_LENGTH} by {RECURSION_DEPTH}'
PROMPT_NICKNAME = 'recursion_prompt'
INITIAL_PROMPT = "i wonder if the ability to recursively improve upon the present is the key to unlocking the boundless potential of the future, a tool of the gods, the engine of progress, the ultimate weapon in the battle against entropy."


def generate_response_parallel(prompt: str) -> str:
    try:
        return ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": TEMP})['response']
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return ""

def generate_chain_parallel(seed_prompt: str, chain_length: int) -> List[str]:
    chain = [seed_prompt]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate_response_parallel, chain[-1]) for _ in range(chain_length)]
        for future in concurrent.futures.as_completed(futures):
            response = future.result()
            if response:
                chain.append(response)
            else:
                break
    return chain

def generate_tree_parallel(seed_prompt: str, chain_length: int, recursion_depth: int, current_depth: int = 1) -> Dict[str, Any]:
    chain = generate_chain_parallel(seed_prompt, chain_length)
    tree = {"prompt": seed_prompt, "responses": chain[1:]}
    
    if current_depth < recursion_depth:
        tree["children"] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_response = {executor.submit(generate_tree_parallel, response, chain_length, recursion_depth, current_depth + 1): response for response in chain[1:]}
            for future in tqdm(concurrent.futures.as_completed(future_to_response), total=len(future_to_response), desc=f"Recursion depth {current_depth}/{recursion_depth}", leave=False):
                child_tree = future.result()
                tree["children"].append(child_tree)
    
    return tree

def save_tree(tree: Dict[str, Any], metadata: Dict[str, Any], filename: Optional[str] = None):
    full_tree = {
        "metadata": metadata,
        "content": tree
    }
    
    if filename is None:
        filename = f'./responses/parallel_{metadata["model_name"]}_at_{metadata["timestamp"]}.json'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(full_tree, f, indent=2)
        

metadata = {
        "tree_key": f'{PROMPT_NICKNAME}_{MODEL_NAME}',
        "timestamp": TIME_STAMP,
        "shape": SHAPE,
        "model_name": MODEL_NAME,
        "chain_length": CHAIN_LENGTH,
        "recursion_depth": RECURSION_DEPTH
    }
tree = generate_tree_parallel(INITIAL_PROMPT, CHAIN_LENGTH, RECURSION_DEPTH)
save_tree(tree, metadata)
