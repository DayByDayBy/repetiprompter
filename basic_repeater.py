import ollama
import json
from datetime import datetime
import tqdm
import numpy as np

MODEL = 'stablelm2:zephyr'

INITIAL_PROMPT = "the ability to recursively improve upon the present is the key to unlocking the boundless potential of the future, a tool of the gods, the engine of progress, the ultimate weapon in the battle against entropy."
PREFIX = "consider the text provided, and then rephrase it as a question, retaining as much of the original meaning as possible. there is no need to reference the instructions in your answer.  the text:    "
SUFFIX = " ---  please now consider the text provided, and then rephrase it as a question. do not refer to these instructions in your answer"

CHAIN_LENGTH = 4
TEMP_MIN = 0.5
TEMP_MAX = 1.0

OUTPUT_PATH = f"basic_rep/{MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

METADATA = {
    "initial_prompt": INITIAL_PROMPT,
    'model': MODEL,
    "recursion_depth": CHAIN_LENGTH,
}

def generate_response(model, current_prompt, temperature):
    try:
        response = ollama.generate(model=model, 
                                prompt=current_prompt, 
                                options={
                                    'temperature': temperature})
        return response['response']
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

with open(OUTPUT_PATH, 'w') as f:
    json.dump(METADATA, f)
    f.flush()
    prompt = INITIAL_PROMPT
    for i in tqdm.tqdm(range(CHAIN_LENGTH)):
        temperature = TEMP_MIN + (TEMP_MAX - TEMP_MIN) * i / (CHAIN_LENGTH - 1)
        current_prompt = prompt
        response = generate_response(MODEL, current_prompt, temperature)
        if response is not None:
            current_prompt = response
            output_data = {
                INITIAL_PROMPT:{
                f'iteration{i}':{
                    'response': response,
                    'temperature': temperature
                    }
                }
            }
            with open(OUTPUT_PATH, 'a') as f:
                json.dump(output_data, f)
                f.write("\n")
                f.flush()
        else:
            break