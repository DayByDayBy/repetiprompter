import ollama
from datetime import datetime
import tqdm
from json_stream_utils import MLOutputWriter


MODEL = 'stablelm2:zephyr'
INITIAL_PROMPT = "the ability to recursively improve upon the present is the key to unlocking the boundless potential of the future, a tool of the gods, the engine of progress, the ultimate weapon in the battle against entropy."
CHAIN_LENGTH = 4
TEMP_MIN = 0.5
TEMP_MAX = 1.0

writer = MLOutputWriter(
    base_path="basic_rep",
    model_name=MODEL
)

metadata = {
    "initial_prompt": INITIAL_PROMPT,
    "model": MODEL,
    "recursion_depth": CHAIN_LENGTH,
    "temperature_range": {"min": TEMP_MIN, "max": TEMP_MAX}
}

output_path = writer.init_experiment(metadata)


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


prompt = INITIAL_PROMPT
for i in tqdm.tqdm(range(CHAIN_LENGTH)):
    temperature = TEMP_MIN + (TEMP_MAX - TEMP_MIN) * i / (CHAIN_LENGTH - 1)
    current_prompt = prompt
    response = generate_response(MODEL, current_prompt, temperature)
    
    if response is not None:
        writer.append_iteration(
            output_path,
            {
                'prompt': current_prompt,
                'response': response,
                'temperature': temperature
            },
            iteration_num=i
        )
        prompt = response
    else:
        break

# Finalize the experiment
writer.finalize_experiment(output_path)