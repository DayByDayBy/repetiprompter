import ollama
from typing import List, Dict
import json

def generate_response_chain(
    initial_prompt: str,
    model: str = 'stablelm2:zephyr',
    chain_length: int = 3
) -> Dict:
    """
    generate chain of responses using ollama, where each response builds on the previous.
    
    args:
        initial_prompt: seed prompt
        model: Ollama model to use
        chain_length: num of responses to generate
    
    returns:
        dict containing the initial prompt and chain of responses
    """
    responses = []
    current_prompt = initial_prompt
    
    try:
        for _ in range(chain_length):
            response = ollama.generate(model=model, prompt=current_prompt)
            response_text = response['response']
            responses.append(response_text)
            # Use the previous response as the next prompt
            current_prompt = response_text
            
        return {
            'initial_prompt': initial_prompt,
            'responses': responses
        }
    
    except Exception as e:
        print(f"Error generating responses: {e}")
        return None

INITIAL_PROMPT = """
the ability to recursively improve upon the present is the key to unlocking 
the boundless potential of the future, a tool of the gods, the engine of progress, 
the ultimate weapon in the battle against entropy.
"""

if __name__ == "__main__":
    result = generate_response_chain(
        initial_prompt=INITIAL_PROMPT,
        model='stablelm2:zephyr',
        chain_length=3
    )
    
    if result:
        print(json.dumps(result, indent=2))