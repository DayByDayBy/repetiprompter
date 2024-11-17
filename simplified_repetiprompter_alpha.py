import ollama
import json
from datetime import datetime
import tqdm
import numpy as np

INITIAL_PROMPT = "..."  # placeholder for user input
TEMP_RANGE = (0.5, 1.0)  # temperature range to increase over time
RECURSION_DEPTH = 5  # number of times to repeat the response chain

OUTPUT_FILE = f"response_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
METADATA = {
    "initial_prompt": INITIAL_PROMPT,
    "temperature_range": TEMP_RANGE,
    "recursion_depth": RECURSION_DEPTH,
}

def generate_response(prompt, temperature):
    response = ollama.generate(prompt, options={'temperature': temperature})
    return response

with open(OUTPUT_FILE, "w") as f:
    json.dump(METADATA, f)
    for i in tqdm.tqdm(range(RECURSION_DEPTH)):
        temperature = np.interp(i, range(RECURSION_DEPTH), TEMP_RANGE)
        response = generate_response(INITIAL_PROMPT, temperature)
        output_data = {
            "prompt": INITIAL_PROMPT,
            "response": response,
            "temperature": temperature,
            "iteration": i,
        }
        with open(OUTPUT_FILE, "a") as f:
            json.dump(output_data, f)
            f.write("\n")