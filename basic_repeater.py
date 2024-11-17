import ollama
import json
from datetime import datetime
import tqdm
import numpy as np

MODEL = 'stablelm2:zephyr'

INITIAL_PROMPT = "the ability to recursively improve upon the present is the key to unlocking the boundless potential of the future, a tool of the gods, the engine of progress, the ultimate weapon in the battle against entropy."
PREFIX = "consider the text provided, and then rephrase it as a question, retaining as much of the original meaning as possible. there is no need to reference the instructions in your answer.  the text:    "
SUFFIX = " ---  please now consider the text provided, and then rephrase it as a question. do not refer to these instructions in your answer"

CHAIN_LENGTH = 100
TEMP_MIN = 0.5
TEMP_MAX = 1.0

OUTPUT_PATH = f"simple_rep/{MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

RESPONSES = []

METADATA = {
    "initial_prompt": INITIAL_PROMPT,
    "temperature_range": (TEMP_MIN, TEMP_MAX),
    "recursion_depth": CHAIN_LENGTH,
    'responses': RESPONSES
}


def generate_response(model, prompt, temperature):
    responses = []
    for i in range(CHAIN_LENGTH):
        try:
            response = ollama.generate(model=model, 
                                    prompt=prompt, 
                                    options={
                                        'temperature': temperature})
            responses.append(response['response'])
            prompt = response
        except Exception as e:
            print(f"Error generating response: {e}")
            break
    return responses

with open(OUTPUT_PATH, 'w') as f:
    json.dump(METADATA, f)
    f.flush()
    for i in tqdm.tqdm(range(CHAIN_LENGTH)):
        temperature = TEMP_MIN + (TEMP_MAX - TEMP_MIN) * i / (CHAIN_LENGTH - 1)
        response = generate_response(MODEL, INITIAL_PROMPT, temperature)
        output_data = {
            'prompt': INITIAL_PROMPT,
            f'response{i}': response,
            'temperature': temperature,
            'iteration': i
        }
        with open(OUTPUT_PATH, 'a') as f:
            json.dump(output_data, f)
            f.write("\n")
            f.flush()