import ollama
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import os
from tqdm import tqdm
import logging
import time
import tiktoken
import random

os.environ['OLLAMA_NUM_PARALLEL'] = '6'

logging.basicConfig(filename='tree_generation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

TIME_STAMP = datetime.now().strftime("%Y%m%d_%H%M")
MODEL_NAME = 'stablelm2:zephyr'
CHAIN_LENGTH = 3
RECURSION_DEPTH = 3
BASE_TEMP = 0.1
MAX_TEMP = 1.00
SHAPE = f'{CHAIN_LENGTH} by {RECURSION_DEPTH}'
PROMPT_NICKNAME = 'recursion_prompt'
# INITIAL_PROMPT = "the ability to recursively improve upon the present is the key to unlocking the boundless potential of the future, a tool of the gods, the engine of progress, the ultimate weapon in the battle against entropy."
# INITIAL_PROMPT = "systems have sub-systems and sub-systems have sub-systems and so on ad infinitum, which is why we're always starting over."
# INITIAL_PROMPT = "terrified of being alone, yet afraid of intimacy, we experience widespread feelings of emptiness, of disconnection, of the unreality of self. and here the computer, a companion without emotional demands, offers a compromise. You can be a loner, but never alone. You can interact, but need never feel vulnerable to another person."
# INITIAL_PROMPT = "as machines become more and more efficient and perfect, so it will become clear that imperfection is the greatness of man."
# INITIAL_PROMPT = "the single biggest problem in communication is the illusion that it has taken place."
INITIAL_PROMPT =  '"positive feed-back increases the gain of the amplifier, negative feed-back reduces it." discuss this idea in terms of gradients and machine learning'
# INITIAL_PROMPT = "a feedback loop is a process in which the outputs of a system are circled back and used as inputs."
PREFIX = "we have been translating an ancient text as best we can with our understanding of their language.  the source is unfortunately broken, and out of order. we recently translated this fragment, and would like you to hypothesise what messages might precede and succeed this one: \n\n"
SUFFIX = "\n\n   what do you think? do you think you can figure out the messages before and after this one?"

# tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-4")



def calculate_total_responses(chain_length: int, recursion_depth: int) -> int:
    return sum(chain_length**i for i in range(recursion_depth + 1)) - 1
def calculate_temp(current_response: int, total_responses: int, base_temp: float, max_temp: float) -> float:
    return base_temp + (max_temp - base_temp) * (current_response / total_responses)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def generate_response(prompt: str, TEMP: float) -> tuple[str, float]:
    start_time = time.time()
    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt, options={'temperature': TEMP})['response']
        end_time = time.time()
        return response, end_time - start_time
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        end_time = time.time()
        return "no response received - check the model's local availability", end_time - start_time

def generate_chain(seed_prompt: str, chain_length: int, current_response: int, total_responses: int) -> List[Dict[str, Any]]:
    chain = [{"text": f'{PREFIX} \n {seed_prompt} \n {SUFFIX}', "tokens": count_tokens(seed_prompt), "generation_time": 0, 'temp': BASE_TEMP}]  
    for i in tqdm(range(chain_length), desc="generating chain", leave=False):
        temp = calculate_temp(current_response + i, total_responses, BASE_TEMP, MAX_TEMP)
        response, gen_time = generate_response(chain[-1]["text"], temp)
        if response:
            chain.append({"text": response, 
                          "tokens": count_tokens(response), 
                          "generation_time": gen_time,
                          "temp": temp})
            current_response += 1
        else:
            break
    return chain



def generate_tree(seed_prompt: str, chain_length: int, current_depth: int, max_depth: int, current_response: int, total_responses: int) -> Dict[str, Any]:
    chain = generate_chain(seed_prompt, chain_length, current_response, total_responses)
    tree = {"prompt": chain[0], "responses": chain[1:]}  
    current_response += len(chain)-1
    
    if current_depth < max_depth:
        tree["children"] = []
        for response in tqdm(chain[1:], desc=f"recursion depth {current_depth}", leave=False):
            child_tree = generate_tree(response["text"], chain_length, current_depth + 1, max_depth, current_response, total_responses)
            current_response += calculate_total_responses(chain_length, max_depth - current_depth - 1)
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
        filename = f'./responses/epsilon_{metadata["model_name"]}_at_{metadata["timestamp"]}.json'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(full_tree, f, indent=2)

if __name__ == '__main__':
    start_time = time.time()
    
    print(f'\n\n running {MODEL_NAME} model \n shape: {SHAPE} \n started: {TIME_STAMP}\n')
    
    total_responses = calculate_total_responses(CHAIN_LENGTH, RECURSION_DEPTH)
    
    metadata = {
        "tree_key": f'{PROMPT_NICKNAME}_{MODEL_NAME}',
        "timestamp": TIME_STAMP,
        "shape": SHAPE,
        "model_name": MODEL_NAME,
        "chain_length": CHAIN_LENGTH,
        "recursion_depth": RECURSION_DEPTH,
        "ollama_num_parallel": os.environ['OLLAMA_NUM_PARALLEL'],
        "total_responses": total_responses
    }
    
    tree = generate_tree(INITIAL_PROMPT, CHAIN_LENGTH, current_depth = 1, max_depth = RECURSION_DEPTH, current_response=0, total_responses=total_responses)
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
