# Repetiprompter: A Conversational AI Experiment in Iterative Feedback

Repetiprompter is a Python utility library that explores the creative potential of iterative feedback in conversational AI. By using responses to prompts as prompts, it generates a series of outputs that evolve over time, inspired by the Alvin Lucier experiment "I Am Sitting in a Room". This process can lead to unexpected transformations, patterns, and insights, offering a unique window into the behavior of Large Language Models (LLMs).

As a 'work in progress', many of the scripts may be incomplete, or have been renderedd obsolete by later version. They've been reatined both for reference and as a (partial) guide to the progress of the idea.

## Features and Functionality

Repetiprompter versions variously:

- Generate responses to prompts using the generate_response function
- Create chains of responses based on a seed prompt and chain length using the generate_chain function
- Save tree data structures to JSON files using the save_tree function
- Manage metadata, storing it in the metadata variable
- Provide utility variables such as start_time, end_time, and TREE



## libraries used:

```
numpy 
pandas
json
ollama 

tqdm

typing
json
datetime
time
os

logging
tiktoken
random
```

You will also need ollama to be installed and running ( see [ollama.com](https://ollama.com) ).




----------------------------------------------------------------



 <br>
 <br>

Feel free to reach out to for help or feedback on using Repetiprompter in your experiments, or even just to ask what's going on