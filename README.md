# repetiprompter

a project based around a fairly simple idea, namely 

#### prompt -> response, where each response becomes the next prompt

the implementation goes a little further than that, but that is the core of the project


## where to look

there are a few scripts, examples of a few approaches, but the most current implementation is `repetiprompter.py` 

### Usage

```bash
# Run with a config file
uv run python repetiprompter.py run configs/your_config.yaml

# List available framing strategies
uv run python repetiprompter.py strategies

# List available Ollama models
uv run python repetiprompter.py models
```

### Modular Architecture

The core functionality is in `modular_rep_set/`:
- `models.py` - Pydantic schemas for config and output
- `config_loader.py` - YAML config loading with validation
- `runner.py` - Unified runner for chain and tree topologies
- `ollama_interface.py` - Ollama client with native token counting
- `framing_strategies.py` - Pluggable prompt framing (simple, liar_paradox, discussion, rephrase, echo, custom)
- `reminder.py` - Probabilistic prompt reinjection
- `temperature_regime.py` - Static, ramp, and schedule temperature control
- `output_writer.py` - Streaming JSONL output
- `embedding_backend.py` - Pluggable embeddings (Ollama, sentence-transformers, HF)

### Analysis Tools

In `json_analyzers/`:
- `semantic_drift.py` - Analyze semantic drift from root/parent nodes using embeddings

```bash
# Analyze a run
uv run python json_analyzers/semantic_drift.py runs/your_run.jsonl --summary-only
```

### older implementations:

```
repetiprompter_delta.py
repetiprompter_gamma.py
repetiprompter_beta.py
parallel_repetiprompter.py
repetiprompter.ipynb

```

those all create *prompt->response* chains of the length 'CHAIN_LENGTH', and then reuse each element in the chain as a seed-prompt for another chain, each of which is used as a seed-chain for another, and so on, 'RECURSION_DEPTH' times

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

Core dependencies (managed via UV):

```
ollama
pydantic
pyyaml
typer
numpy
```

Legacy scripts also use:

```
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