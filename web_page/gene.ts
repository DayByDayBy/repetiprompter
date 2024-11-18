import ollama from 'ollama';

async function ollamaGenerate(model: string, prompt: string, suffix?: string): Promise<string> {
    const response = await ollama.generate({
        model,
        prompt,
        suffix,
        stream: false,
    });

    return response.response;
}



let seed_prompt: string = "the ability to recursively improve upon the present is the key to unlocking the boundless potential of the future, a tool of the gods, the engine of progress, the ultimate weapon in the battle against entropy."
// let suffix: string = "   -   please consider the text provided, and then rephrase it as a question, retaining as much of the original meaning as possible. there is no need to reference these instructions in your answer"

ollamaGenerate('llama3', seed_prompt)
    .then((result) => console.log('generated response:', result))
    .catch((error) => console.error('error:', error));
