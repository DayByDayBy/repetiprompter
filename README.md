# repetiprompter

a project based around a fairly simple idea, namely 

- prompt -> response where response becomes prompt

the implementation goes a little further than that, but that is the core of the project


## where to look

there are a few scripts, examples or a few approaches, but the most current implementations are 

```
cubic_repetiprompter_beta.py
parallel_repetiprompter.py
repetiprompter.ipynb
```
the first of those creates prompts chains as described above of the length 'CHAIN_LENGTH', and then reuses each element in the chain as a seed prompt for another chain, each of which is used as a seed for another chain, and so on, 'RECURSION_DEPTH' times

the second of those does similar, but does so employing some parallelisation via the 'concurrent' library

the third is a jupyter notbeook refactoring, with some basic visualisations and stats information


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


