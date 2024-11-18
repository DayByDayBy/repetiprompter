import ollama from 'ollama';

let seed_prompt: string = "the ability to recursively improve upon the present is the key to unlocking the boundless potential of the future, a tool of the gods, the engine of progress, the ultimate weapon in the battle against entropy."

async function createChain(prompt: string): Promise<{ prompt: string; response: string }[]> {
    const chain: { prompt: string; response: string }[] = [];
    let currentPrompt = prompt;

    while (currentPrompt !== '') {
        const response = await generateResponse('llama3', currentPrompt);
        chain.push({ prompt: currentPrompt, response });
        currentPrompt = response;
    }

    return chain;
}

async function generateResponse(model: string, prompt: string, suffix?: string): Promise<string> {
    return ollama.generate({
        model,
        prompt,
        suffix,
    }).then((response) => response.response);
}
console.log(generateResponse('llama3', seed_prompt));

export { createChain, generateResponse };