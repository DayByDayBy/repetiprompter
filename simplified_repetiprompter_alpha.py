import ollama

CHAIN_LENGTH = 10
MODEL = 'stablelm2:zephyr'
INITIAL_PROMPT = "the ability to recursively improve upon the present is the key to unlocking the boundless potential of the future, a tool of the gods, the engine of progress, the ultimate weapon in the battle against entropy."

responses = []

for i in range(CHAIN_LENGTH):
    response = ollama.generate(model=MODEL, prompt=INITIAL_PROMPT)['response']
    responses.append(response)

print(responses)