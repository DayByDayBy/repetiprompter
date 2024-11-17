import ollama
from typing import List
from datetime import datetime

time_stamp = datetime.now().strftime("%Y%m%d_%H%M")


MODEL_NAME = 'llama2'
DEPTH = 10
INITIAL_PROMPT = "consider: the ability to recursively improve upon the present is the key to unlocking the boundless potential of the future, a tool of the gods, the engine of progress, the ultimate weapon in the battle against entropy."

def generate_recursive(prompt: str, depth: int = DEPTH) -> List[str]:
    if depth == 0:
        return [prompt]
    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt)['response']
        return [prompt] + generate_recursive(response, depth - 1)
    except Exception as e:
        print(f"Error generating response: {e}")
        return [prompt]

if __name__ == '__main__':
    
    responses = generate_recursive(INITIAL_PROMPT)
    print(responses)
    
    with open('responses.txt', 'w') as f:
        f.write('\n\n'.join(responses))