import { createChain, generateResponse } from './repetiprompter';

document.getElementById('use-seed-prompt')!.addEventListener('click', () => {
    const seedPrompt = document.getElementById('seed-prompt')!.value;
    const chain = createChain(seedPrompt);
    displayChain(chain);
});

document.getElementById('generate-prompt')!.addEventListener('click', () => {
    const prompt = 'This is a generated prompt';
    const chain = createChain(prompt);
    displayChain(chain);
});

function displayChain(chain: { prompt: string; response: string }[]) {
    const chainList = document.getElementById('chain-list')!;
    chainList.innerHTML = '';

    chain.forEach((item, index) => {
        const listItem = document.createElement('li');
        listItem.className = 'chain-item';
        listItem.innerHTML = `${item.prompt} -> ${item.response}`;

        if (index > 0) {
            const previousItem = chain[index - 1];
            listItem.previousElementSibling!.classList.add('response-item');
        }

        chainList.appendChild(listItem);
    });
}

function displayResponse(response: string) {
    const responseList = document.getElementById('response-list')!;
    responseList.innerHTML = '';

    const listItem = document.createElement('li');
    listItem.className = 'response-item';
    listItem.innerHTML = response;

    responseList.appendChild(listItem);
}