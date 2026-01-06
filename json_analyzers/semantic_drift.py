#!/usr/bin/env python3
"""
Semantic drift analyzer for JSONL runs.

Computes embeddings for all nodes in a run, calculates cosine distance 
to the root and parent nodes, and exports results for analysis.

Extended metrics:
- drift_local: Distance to previous response (step-wise)
- drift_global: Distance to run centroid
- drift_instruction: Distance to original prompt

Embedding modes:
- prompt_only: Embed only prompt text
- response_only: Embed only response text  
- prompt_response: Embed concatenated prompt + response (default)
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from modular_rep_set.embedding_backend import (
    cosine_distance,
    create_embedding_backend,
    EmbeddingBackend,
)

from .core.tree_utils import (
    parse_node_id,
    get_depth,
    load_jsonl_nodes,
)


# parse_node_id, get_depth, load_jsonl_nodes imported from core.tree_utils


def extract_text_for_embedding(
    node: Dict,
    mode: str = "prompt_response",
) -> str:
    """Extract text content from node for embedding.
    
    Args:
        node: Node dictionary
        mode: Embedding mode - "prompt_only", "response_only", or "prompt_response"
        
    Returns:
        Text to embed
    """
    prompt = node.get('prompt', '')
    response = node.get('response', '')
    
    if mode == "prompt_only":
        return prompt.strip()
    elif mode == "response_only":
        return response.strip()
    else:  # prompt_response
        return f"{prompt} {response}".strip()


def analyze_semantic_drift(
    jsonl_file: Path,
    backend: EmbeddingBackend,
    output_csv: Optional[Path] = None,
    embedding_mode: str = "prompt_response",
    extended: bool = True,
) -> List[Dict]:
    """Analyze semantic drift in a JSONL run file.
    
    Args:
        jsonl_file: Path to JSONL file
        backend: Embedding backend instance
        output_csv: Optional path to save CSV results
        embedding_mode: Mode for text extraction (prompt_only, response_only, prompt_response)
        extended: Include extended metrics (drift_local, drift_global, drift_instruction)
    
    Returns:
        List of drift analysis results
    """
    print(f"Loading nodes from {jsonl_file}...")
    nodes = load_jsonl_nodes(jsonl_file)
    print(f"Loaded {len(nodes)} nodes")
    
    # Sort nodes by ID to ensure proper order
    nodes.sort(key=lambda n: parse_node_id(n['node_id']))
    
    # Store embeddings by node ID
    embeddings: Dict[str, np.ndarray] = {}
    
    # For extended metrics
    centroid_sum: Optional[np.ndarray] = None
    centroid_count = 0
    original_prompt_embedding: Optional[np.ndarray] = None
    previous_embedding: Optional[np.ndarray] = None
    
    # Results storage
    results = []
    
    print(f"Computing embeddings (mode: {embedding_mode})...")
    for i, node in enumerate(nodes):
        node_id = node['node_id']
        text = extract_text_for_embedding(node, embedding_mode)
        
        if not text:
            print(f"Warning: No text content for node {node_id}")
            continue
        
        # Compute embedding
        embedding = backend.encode(text)
        embeddings[node_id] = embedding
        
        # Update centroid incrementally
        if centroid_sum is None:
            centroid_sum = embedding.copy()
        else:
            centroid_sum = centroid_sum + embedding
        centroid_count += 1
        
        # Store original prompt embedding (from root node)
        depth = get_depth(node_id)
        if depth == 0 and extended:
            prompt_text = node.get('prompt', '')
            if prompt_text:
                original_prompt_embedding = backend.encode(prompt_text)
        
        parent_id = node.get('parent_id')
        
        # Drift from root (if not root)
        drift_from_root = None
        if depth > 0:
            root_embedding = None
            for node_id_check, emb_check in embeddings.items():
                if get_depth(node_id_check) == 0:
                    root_embedding = emb_check
                    break
            
            if root_embedding is not None:
                drift_from_root = cosine_distance(embedding, root_embedding)
        
        # Drift from parent
        drift_from_parent = None
        if parent_id:
            parent_embedding = embeddings.get(parent_id)
            if parent_embedding is None:
                if '.' in node_id:
                    parts = node_id.split('.')
                    if '_' in parts[0] and len(parts[0]) > 10:
                        parent_id_alt = '.'.join(parts[:-1])
                        parent_embedding = embeddings.get(parent_id_alt)
            
            if parent_embedding is not None:
                drift_from_parent = cosine_distance(embedding, parent_embedding)
        
        result = {
            'node_id': node_id,
            'depth': depth,
            'step': i,
            'drift_from_root': drift_from_root,
            'drift_from_parent': drift_from_parent,
            'prompt_length': len(node.get('prompt', '')),
            'response_length': len(node.get('response', '')),
        }
        
        # Extended metrics
        if extended:
            # drift_local: distance to previous response (step-wise)
            drift_local = None
            if previous_embedding is not None:
                drift_local = cosine_distance(embedding, previous_embedding)
            result['drift_local'] = drift_local
            
            # drift_global: distance to run centroid
            drift_global = None
            if centroid_count > 0:
                centroid = centroid_sum / centroid_count
                drift_global = cosine_distance(embedding, centroid)
            result['drift_global'] = drift_global
            
            # drift_instruction: distance to original prompt
            drift_instruction = None
            if original_prompt_embedding is not None:
                drift_instruction = cosine_distance(embedding, original_prompt_embedding)
            result['drift_instruction'] = drift_instruction
            
            result['embedding_mode'] = embedding_mode
        
        results.append(result)
        previous_embedding = embedding
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(nodes)} nodes")
    
    # Save to CSV if requested
    if output_csv:
        save_results_to_csv(results, output_csv, extended=extended)
        print(f"Results saved to {output_csv}")
    
    return results


def save_results_to_csv(results: List[Dict], output_path: Path, extended: bool = True):
    """Save drift analysis results to CSV."""
    fieldnames = [
        'node_id',
        'depth', 
        'step',
        'drift_from_root',
        'drift_from_parent',
        'prompt_length',
        'response_length',
    ]
    
    if extended:
        fieldnames.extend([
            'drift_local',
            'drift_global',
            'drift_instruction',
            'embedding_mode',
        ])
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary_stats(results: List[Dict]):
    """Print summary statistics of drift analysis."""
    if not results:
        print("No results to analyze")
        return
    
    # Filter out None values for statistics
    root_drifts = [r['drift_from_root'] for r in results if r['drift_from_root'] is not None]
    parent_drifts = [r['drift_from_parent'] for r in results if r['drift_from_parent'] is not None]
    
    print("\n=== Semantic Drift Analysis Summary ===")
    print(f"Total nodes analyzed: {len(results)}")
    print(f"Max depth: {max(r['depth'] for r in results)}")
    
    if root_drifts:
        print(f"\nDrift from root:")
        print(f"  Mean: {np.mean(root_drifts):.4f}")
        print(f"  Std:  {np.std(root_drifts):.4f}")
        print(f"  Min:  {np.min(root_drifts):.4f}")
        print(f"  Max:  {np.max(root_drifts):.4f}")
    
    if parent_drifts:
        print(f"\nDrift from parent:")
        print(f"  Mean: {np.mean(parent_drifts):.4f}")
        print(f"  Std:  {np.std(parent_drifts):.4f}")
        print(f"  Min:  {np.min(parent_drifts):.4f}")
        print(f"  Max:  {np.max(parent_drifts):.4f}")
    
    # Group by depth
    depth_stats = {}
    for result in results:
        depth = result['depth']
        if depth not in depth_stats:
            depth_stats[depth] = []
        if result['drift_from_root'] is not None:
            depth_stats[depth].append(result['drift_from_root'])
    
    print(f"\nDrift from root by depth:")
    for depth in sorted(depth_stats.keys()):
        drifts = depth_stats[depth]
        if drifts:  # Only print if we have data
            print(f"  Depth {depth}: mean={np.mean(drifts):.4f}, n={len(drifts)}")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Analyze semantic drift in JSONL runs")
    parser.add_argument("jsonl_file", type=Path, help="Path to JSONL run file")
    parser.add_argument(
        "--backend", 
        choices=["ollama", "sentence-transformers", "huggingface"],
        default="ollama",
        help="Embedding backend to use"
    )
    parser.add_argument("--model", help="Model name for embedding backend")
    parser.add_argument(
        "--mode",
        choices=["prompt_only", "response_only", "prompt_response"],
        default="prompt_response",
        help="Embedding mode: what text to embed"
    )
    parser.add_argument(
        "--no-extended",
        action="store_true",
        help="Disable extended metrics (drift_local, drift_global, drift_instruction)"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        help="Output CSV file (default: <jsonl_file>_drift.csv)"
    )
    parser.add_argument(
        "--summary-only", 
        action="store_true", 
        help="Only print summary, don't save CSV"
    )
    
    args = parser.parse_args()
    
    if not args.jsonl_file.exists():
        print(f"Error: File {args.jsonl_file} does not exist")
        sys.exit(1)
    
    # Set output path if not provided
    if not args.output and not args.summary_only:
        suffix = "_drift_ext.csv" if not args.no_extended else "_drift.csv"
        args.output = args.jsonl_file.with_name(f"{args.jsonl_file.stem}{suffix}")
    
    # Create embedding backend
    print(f"Creating embedding backend: {args.backend}")
    backend = create_embedding_backend(args.backend, args.model)
    
    # Run analysis
    results = analyze_semantic_drift(
        args.jsonl_file, 
        backend, 
        args.output if not args.summary_only else None,
        embedding_mode=args.mode,
        extended=not args.no_extended,
    )
    
    # Print summary
    print_summary_stats(results)


if __name__ == "__main__":
    main()
