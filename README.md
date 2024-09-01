# repetiprompter

a project based around a fairly simple idea, namely 

#### prompt -> response, where each response becomes the next prompt

the implementation goes a little further than that, but that is the core of the project


## where to look

there are a few scripts, examples of a few approaches, but the most current implementations are: 



```
repetiprompter_delta.py

repetiprompter_gamma.py

repetiprompter_beta.py
parallel_repetiprompter.py
repetiprompter.ipynb

```
they all create *prompt->response* chains of the length 'CHAIN_LENGTH', and then reuse each element in the chain as a seed-prompt for another chain, each of which is used as a seed-chain for another, and so on, 'RECURSION_DEPTH' times

the parallel one does so, employing some parallelisation via the 'concurrent' library

the last one is a jupyter notbeook refactoring of an earlier iteration, with some basic visualisations and stats information


be sure to check the values for CHAIN_LENGTH and RECURSION_DEPTH before running, (in particular the latter) as it will take a while to complete if those are much above 4 and 4, especially for larger models 

check how many responses will be generated with this:

```
chain_length = CHAIN_LENGTH
recursion_depth = RECURSION_DEPTH

total_chains = sum(chain_length ** i for i in range(recursion_depth))
total_prompt_response_actions = total_chains * chain_length


print(total_chains, total_prompt_response_actions)
```

(the same code is in a cell of the notebook)

bear in mind model choice etc will determine how long each prompt->response cycle will take


## requirements

using the following libraries:

```

ollama
typing (Dict, List, Any, Optional)
json
datetime
os
tqdm
logging

pandas 
matplotlib 
seaborn 
networkx 
wordcloud

```

these are used by the parallel-repetiprompter:
```
concurrent.futures
functools (partial)
```


some models used:

```

dolphin-mistral
dolphin-mixtral
gemma2:27b
llama2
llama3
llama3.1:8b
llama3.1:70b
mistral
mixtral
moondream
openchat
openhermes
orca-mini
phi
phi3
phi3:14b
stablelm2:zephyr
tinyllama
wizardlm2
zephyr

```