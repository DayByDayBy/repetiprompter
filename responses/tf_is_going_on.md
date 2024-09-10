# Recursive Prompt-Response Tree Generator

This script generates a tree of prompt-response chains using the Ollama language model. It creates a hierarchical structure of prompts and responses, with each response potentially becoming a new prompt for further exploration.

## Dependencies

- ollama
- typing
- json
- datetime
- os
- tqdm
- logging

## Configuration

The script uses several global variables for configuration:

- `TIME_STAMP`: Current date and time in "YYYYMMDD_HHMM" format.
- `MODEL_NAME`: Name of the Ollama model to use (default: 'tinyllama' for ease of testing).
- `CHAIN_LENGTH`: Number of responses in each chain (default: 2).
- `RECURSION_DEPTH`: Number of levels in the response tree (default: 2).
- `SHAPE`: A string representation of the tree shape (e.g., "2 by 2").
- `PROMPT_NICKNAME`: A nickname for the initial prompt (default: 'recursion_prompt').
- `INITIAL_PROMPT`: The seed prompt to start the tree generation.

## Functions

### generate_response(prompt: str) -> str

Generates a single response for a given prompt using the Ollama model.

- **Parameters:**
  - `prompt` (str): The input prompt.
- **Returns:**
  - str: The generated response, or an empty string if an error occurs.

### generate_chain(seed_prompt: str, chain_length: int) -> List[str]

Generates a chain of responses starting from a seed prompt.

- **Parameters:**
  - `seed_prompt` (str): The initial prompt to start the chain.
  - `chain_length` (int): The number of responses to generate in the chain.
- **Returns:**
  - List[str]: A list containing the seed prompt and generated responses.

### generate_tree(seed_prompt: str, chain_length: int, recursion_depth: int) -> Dict[str, Any]

Recursively generates a tree of prompt-response chains.

- **Parameters:**
  - `seed_prompt` (str): The initial prompt to start the tree.
  - `chain_length` (int): The number of responses in each chain.
  - `recursion_depth` (int): The number of levels to recurse.
- **Returns:**
  - Dict[str, Any]: A dictionary representing the tree structure.

### save_tree(tree: Dict[str, Any], metadata: Dict[str, Any], filename: Optional[str] = None)

Saves the generated tree and metadata to a JSON file.

- **Parameters:**
  - `tree` (Dict[str, Any]): The generated tree structure.
  - `metadata` (Dict[str, Any]): Metadata about the tree generation process.
  - `filename` (Optional[str]): The filename to save the JSON. If None, a default name is generated.

## Main Execution

The script's main execution:

1. Sets up metadata for the tree generation process.
2. Generates the tree using the `generate_tree` function.
3. Saves the generated tree and metadata using the `save_tree` function.

## Output

The script generates a JSON file containing:
- Metadata about the generation process.
- The full tree structure of prompts and responses.

The JSON file is saved in the `./responses/` directory with a name format:
`tree_{model_name}_at_{timestamp}.json`

## Error Handling

Errors during response generation are logged to a file named 'tree_generation.log'.

## Progress Tracking

The script uses tqdm to display progress bars for both chain generation and tree recursion, providing visual feedback on the generation process.