import os
import json
import time
from typing import List, Dict
from tqdm import tqdm
import logging
import ollama
from anytree import Node, RenderTree

# logging
logging.basicConfig(filename='tree_generation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# constants
MODEL_NAME = 'stablelm2:zephyr'
CHAIN_LENGTH = 3
RECURSION_DEPTH = 3
total_generations = (sum(CHAIN_LENGTH**i for i in range(RECURSION_DEPTH + 1)) - 1)
TEMPERATURE = 0.67
INITIAL_PROMPT = "The single biggest problem in communication is the illusion that it has taken place."
PREFIX = "i awoke to find george bernard shaw standing by me, saying  "
SUFFIX = " is this a problematic view?"

def generate_response(prompt: str) -> str:
    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt, options={'temperature': TEMPERATURE})
        return response['response']
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "No response received - check if Ollama is running and the model is available."

def generate_chain(seed_prompt: str, chain_length: int) -> List[Dict[str, str]]:
    chain = [{"text": seed_prompt}]
    for _ in tqdm(range(chain_length), desc="Generating chain"):
        prompt = f"{PREFIX}\n{chain[-1]['text']}\n{SUFFIX}"
        response = generate_response(prompt)
        chain.append({"text": response})
    return chain

def generate_tree(seed_prompt: str, chain_length: int, current_depth: int, max_depth: int) -> Dict:
    chain = generate_chain(seed_prompt, chain_length)
    tree = {"prompt": chain[0], "responses": chain[1:]}
    
    if current_depth <= max_depth:
        tree["children"] = []
        for response in chain[1:]:
            child_tree = generate_tree(response["text"], chain_length, current_depth + 1, max_depth)
            tree["children"].append(child_tree)
    
    return tree

def save_tree(tree: Dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(tree, f, indent=2)

def visualize_tree(tree):
    def create_nodes(data, parent=None, node_count=[0]):
        if not data or 'prompt' not in data:
            return None
        node_count[0] += 1
        node_name = f'Node_{node_count[0]}'
        prompt_text = data['prompt'].get('text', 'no text')
        node = Node(prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text, parent=parent)
        for response in data.get('responses', []):
            node_count[0] += 1
            node_name = f'Node_{node_count[0]}'
            response_text = response.get('text', 'no response')
            Node(response_text[:50] + '...' if len(response_text) > 50 else response_text, parent=node)
        for child in data.get('children', []):
            create_nodes(child, node, node_count)
        return node
    
    root = create_nodes(tree)
    if root:
        for pre, _, node in RenderTree(root):
            print(f"{pre}{node.name}")
        else:
            print('unable to visualise tree: invalid tree structure')


def main():
    start_time = time.time()
    
    print(f"Generating response tree using {MODEL_NAME}...")
    tree = generate_tree(INITIAL_PROMPT, CHAIN_LENGTH, current_depth=1, max_depth=RECURSION_DEPTH)
    filename = f'simplified_responses/simplified_{time.strftime("%Y%m%d_%H%M%S")}.json'
    save_tree(tree, filename)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTree generation complete!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Results saved to: {filename}")
    print("\nvisualising tree structure...")

    visualize_tree(tree)

    
    logging.info(f"Run completed. MODEL: {MODEL_NAME}, CHAIN_LENGTH: {CHAIN_LENGTH}, "
                 f"RECURSION_DEPTH: {RECURSION_DEPTH}, Total time: {total_time:.2f}s")

if __name__ == '__main__':
    main()
