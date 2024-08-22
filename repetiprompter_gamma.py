# a lot like the _beta, but with some more metrics/stats being captured 


import ollama
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import os
from tqdm import tqdm
import logging
import time
import tiktoken

os.environ['OLLAMA_NUM_PARALLEL'] = '3000'

logging.basicConfig(filename='tree_generation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

TIME_STAMP = datetime.now().strftime("%Y%m%d_%H%M")
MODEL_NAME = 'llama3.1'
TEMP = 0.67
CHAIN_LENGTH = 5
RECURSION_DEPTH = 5
SHAPE = f'{CHAIN_LENGTH} by {RECURSION_DEPTH}'
PROMPT_NICKNAME = 'recursion_prompt'
INITIAL_PROMPT = "consider: the ability to recursively improve upon the present is the key to unlocking the boundless potential of the future, a tool of the gods, the engine of progress, the ultimate weapon in the battle against entropy."

# tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-4")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def generate_response(prompt: str) -> tuple[str, float]:
    start_time = time.time()
    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": TEMP})['response']
        end_time = time.time()
        return response, end_time - start_time
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        end_time = time.time()
        return "", end_time - start_time

def generate_chain(seed_prompt: str, chain_length: int) -> List[Dict[str, Any]]:
    chain = [{"text": seed_prompt, "tokens": count_tokens(seed_prompt), "generation_time": 0}]
    for _ in tqdm(range(chain_length), desc="generating chain", leave=False):
        response, gen_time = generate_response(chain[-1]["text"])
        if response:
            chain.append({"text": response, "tokens": count_tokens(response), "generation_time": gen_time})
        else:
            break
    return chain

def generate_tree(seed_prompt: str, chain_length: int, recursion_depth: int) -> Dict[str, Any]:
    chain = generate_chain(seed_prompt, chain_length)
    tree = {"prompt": chain[0], "responses": chain[1:]}
    
    if recursion_depth > 1:
        tree["children"] = []
        for response in tqdm(chain[1:], desc=f"recursion depth {recursion_depth}", leave=False):
            child_tree = generate_tree(response["text"], chain_length, recursion_depth - 1)
            tree["children"].append(child_tree)
    
    return tree

def calculate_tree_stats(tree: Dict[str, Any]) -> Dict[str, Any]:
    total_tokens = tree["prompt"]["tokens"] + sum(r["tokens"] for r in tree["responses"])
    total_time = sum(r["generation_time"] for r in tree["responses"])
    node_count = 1 + len(tree["responses"])
    
    if "children" in tree:
        for child in tree["children"]:
            child_stats = calculate_tree_stats(child)
            total_tokens += child_stats["total_tokens"]
            total_time += child_stats["total_time"]
            node_count += child_stats["node_count"]
    
    return {
        "total_tokens": total_tokens,
        "total_time": total_time,
        "node_count": node_count,
        "tokens_per_second": total_tokens / total_time if total_time > 0 else 0
    }

def save_tree(tree: Dict[str, Any], metadata: Dict[str, Any], filename: Optional[str] = None):
    stats = calculate_tree_stats(tree)
    metadata.update(stats)
    
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
    start_time = time.time()
    print(f'\n\nRunning {MODEL_NAME} model\n\n')
    
    metadata = {
        "tree_key": f'{PROMPT_NICKNAME}_{MODEL_NAME}',
        "timestamp": TIME_STAMP,
        "shape": SHAPE,
        "model_name": MODEL_NAME,
        "chain_length": CHAIN_LENGTH,
        "recursion_depth": RECURSION_DEPTH,
        "ollama_num_parallel": os.environ['OLLAMA_NUM_PARALLEL']
    }
    
    tree = generate_tree(INITIAL_PROMPT, CHAIN_LENGTH, RECURSION_DEPTH)
    save_tree(tree, metadata)
    
    end_time = time.time()
    total_execution_time = end_time - start_time
    
    print("\n\ngenerated tree saved.\n\n")
    print(f"Total execution time: {total_execution_time:.2f} seconds")
    print(f"Total tokens generated: {metadata['total_tokens']}")
    print(f"Overall tokens per second: {metadata['total_tokens'] / total_execution_time:.2f}")
    print(f"Total nodes in tree: {metadata['node_count']}")
    
    logging.info(f"Run completed. OLLAMA_NUM_PARALLEL: {os.environ['OLLAMA_NUM_PARALLEL']}, "
                 f"CHAIN_LENGTH: {CHAIN_LENGTH}, RECURSION_DEPTH: {RECURSION_DEPTH}, "
                 f"Total time: {total_execution_time:.2f}s, "
                 f"Tokens/s: {metadata['total_tokens'] / total_execution_time:.2f}")