import ollama

CHAIN_LENGTH = 3
MODEL = 'stablelm2:zephyr'
INITIAL_PROMPT = "the ability to recursively improve upon the present is the key to unlocking the boundless potential of the future, a tool of the gods, the engine of progress, the ultimate weapon in the battle against entropy."

responses = {
    'initial_prompt':INITIAL_PROMPT, 
    'responses':{
        [ollama.generate(model=MODEL, prompt=INITIAL_PROMPT)['response'] for i in range(0, CHAIN_LENGTH)]
    }}

print(responses)




