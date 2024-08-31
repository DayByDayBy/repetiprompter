from ascii_trees import tree_to_ascii
import time

def tree_to_dict(tree):
    """Convert our tree structure to a dictionary format suitable for ascii-trees."""
    node = {
        "name": tree["prompt"]["text"][:30] + "...",  # long prompts
        "children": []
    }
    if "children" in tree:
        for child in tree["responses"]:
            node["children"].append(tree_to_dict(child))
    return node

def visualise_tree(tree):
    """Generate and print an ASCII representation of the tree."""
    tree_dict = tree_to_dict(tree)
    ascii_tree = tree_to_ascii(tree_dict)
    print("\nCurrent Tree Structure:")
    print(ascii_tree)

# generate_tree function that periodically visualises the tree
def generate_tree(seed_prompt: str, chain_length: int, current_depth: int, max_depth: int) -> Dict[str, Any]:
    temp = calculate_temp(current_depth, max_depth, base_temp=BASE_TEMP, max_temp=MAX_TEMP)
    chain = generate_chain(seed_prompt, chain_length, temp)
    tree = {"prompt": chain[0], "responses": chain[1:]}   
    
    if current_depth < max_depth:
        tree["children"] = []
        for response in tqdm(chain[1:], desc=f"recursion depth {current_depth}", leave=False):
            child_tree = generate_tree(response["text"], chain_length, current_depth + 1, max_depth)
            tree["children"].append(child_tree)
        
        # visualise tree every 5 minutes
        if time.time() - generate_tree.last_visualization > 300:  # 300 seconds = 5 minutes
            visualise_tree(tree)
            generate_tree.last_visualization = time.time()
    
    return tree

# initialise last visualization time
generate_tree.last_visualization = time.time()

# for main execution:
tree = generate_tree(INITIAL_PROMPT, CHAIN_LENGTH, current_depth=1, max_depth=RECURSION_DEPTH)
visualise_tree(tree)  # Final visualization of the complete tree