import streamlit as st
import ollama
from typing import Dict, List, Any
import json
from datetime import datetime
import os
import time
import networkx as nx
import matplotlib.pyplot as plt
from le_delta import generate_tree, calculate_temp, save_tree, calculate_tree_stats

# streamlit and tiktoken not oplaying well together, not sure why yet, so workaround:
try:
    import tiktoken
    def count_tokens(text: str) -> int:
        tokenizer = tiktoken.encoding_for_model("gpt-4")
        return len(tokenizer.encode(text))
except ImportError:
    st.warning("tiktoken not found. Using a simple word count instead.")
    def count_tokens(text: str) -> int:
        return len(text.split())
    

def visualize_tree(tree: Dict[str, Any]):
    G = nx.Graph()
    def add_nodes(node, parent=None):
        node_id = id(node)
        G.add_node(node_id, text=node['prompt']['text'][:30] + '...')  # hiding some text for display
        if parent:
            G.add_edge(parent, node_id)
        for child in node.get('children', []):
            add_nodes(child, node_id)   
    add_nodes(tree)
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=8, font_weight='bold')
    nx.draw_networkx_labels(G, pos, {node: G.nodes[node]['text'] for node in G.nodes()})
    
    st.pyplot(plt)
    

def generate_tree_with_updates(initial_prompt, chain_length, current_depth, max_depth):
    progress_bar = st.progress(0)
    status_text = st.empty()
    tree_vis = st.empty()
    
    def update_callback(current_depth, total_nodes, max_nodes):
        progress = current_depth / max_depth
        progress_bar.progress(progress)
        status_text.text(f"Generating tree: Depth {current_depth}/{max_depth}, Nodes: {total_nodes}/{max_nodes}")
        
    tree = generate_tree(initial_prompt, chain_length, current_depth, max_depth, update_callback)
    
    progress_bar.progress(1.0)
    status_text.text("Tree generation complete!")
    visualize_tree(tree)
    return tree


    
def main():
    st.title("Tree Generator Dashboard")

    st.sidebar.header("Parameters")
    model_name = st.sidebar.selectbox("Model", ["llama3.1", "gpt-3.5-turbo", "gpt-4"])
    chain_length = st.sidebar.slider("Chain Length", 1, 10, 2)
    recursion_depth = st.sidebar.slider("Recursion Depth", 1, 5, 2)
    base_temp = st.sidebar.slider("Base Temperature", 0.0, 1.0, 0.01, 0.01)
    max_temp = st.sidebar.slider("Max Temperature", 0.0, 1.0, 1.0, 0.01)
    temp_behavior = st.sidebar.radio("Temperature Behavior", ["Increase with Depth", "Decrease with Depth"])

    initial_prompt = st.text_area("Initial Prompt", "Enter your initial prompt here...")

    if st.button("Generate Tree"):
        if not initial_prompt:
            st.error("Please enter an initial prompt.")
            return

        if temp_behavior == "Increase with Depth":
            def calc_temp(current_depth, max_depth):
                return calculate_temp(current_depth, max_depth, base_temp, max_temp)
        else:
            def calc_temp(current_depth, max_depth):
                return calculate_temp(max_depth - current_depth + 1, max_depth, base_temp, max_temp)

        with st.spinner("Generating tree..."):
            tree = generate_tree(initial_prompt, chain_length, 1, recursion_depth)

        st.subheader("Tree Statistics")
        stats = calculate_tree_stats(tree)
        st.write(f"Total Tokens: {stats['total_tokens']}")
        st.write(f"Total Time: {stats['total_time']:.2f} seconds")
        st.write(f"Node Count: {stats['node_count']}")
        st.write(f"Tokens per Second: {stats['tokens_per_second']:.2f}")

        if st.button("Save Tree"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            metadata = {
                "tree_key": f"streamlit_generated_{model_name}",
                "timestamp": timestamp,
                "shape": f"{chain_length} by {recursion_depth}",
                "model_name": model_name,
                "chain_length": chain_length,
                "recursion_depth": recursion_depth,
                "base_temp": base_temp,
                "max_temp": max_temp,
                "temp_behavior": temp_behavior
            }
            filename = f"./responses/tree_{model_name}_at_{timestamp}.json"
            save_tree(tree, metadata, filename)
            st.success(f"Tree saved to {filename}")

        st.subheader("Generated Tree")
        st.json(tree)

if __name__ == "__main__":
    main()