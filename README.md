# repetiprompter

a project based around a fairly simple idea, namely 

#### prompt -> response, where the response becomes the next prompt

the implementation goes a little further than that, but that is the core of the project



## where to look

there are a few scripts, examples of a few approaches, but the most current implementations are: 


```
repetiprompter_beta.py
parallel_repetiprompter.py
repetiprompter.ipynb
```
the first of those creates prompts chains as described above of the length 'CHAIN_LENGTH', and then reuses each element in the chain as a seed prompt for another chain, each of which is used as a seed for another chain, and so on, 'RECURSION_DEPTH' times

the second does similar, but does so employing some parallelisation via the 'concurrent' library, the third is a jupyter notbeook refactoring, with some basic visualisations and stats information


be sure to check the values for CHAIN_LENGTH and RECURSION_DEPTH before running, 
(in particular the latter) as it will take a while to complete if those are much above 4 and 4 

check how many respponses will be generated with this:

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

all three approaches use the following libraries:
```
ollama
typing (Dict, List, Any, Optional)
json
datetime
os
tqdm
logging
```

these are used by the parallel-repetiprompter:
```
concurrent.futures
functools (partial)
```

these are used by the notebook:

```
pandas 
matplotlib 
seaborn 
networkx 
wordcloud
```


