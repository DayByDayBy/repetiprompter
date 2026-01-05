#!/usr/bin/env python3
"""
Semantic drift analyzer for JSONL runs.

Computes embeddings for all nodes in a run, calculates cosine distance 
to the root and parent nodes, and exports results for analysis.
"""

import argparse
import csv
import json
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


def parse_node_id(node_id: str) -> Tuple[int, ...]:
    """Parse hierarchical node ID into tuple of integers."""
    # Remove run_id prefix if present (e.g., "20260105_140916_7b71.0" -> "0")
    if '.' in node_id:
        parts = node_id.split('.')
        # Check if first part looks like a run_id (contains timestamp)
        first_part = parts[0]
        if '_' in first_part and len(first_part) > 10:
            # This looks like a run_id, extract the hierarchical part
            return tuple(int(part) for part in parts[1:] if part.isdigit())
    
    # Fallback: try to parse all parts as integers
    return tuple(int(part) for part in node_id.split('.') if part.isdigit())


def get_depth(node_id: str) -> int:
    """Get depth from node ID."""
    # Remove run_id prefix if present
    if '.' in node_id:
        parts = node_id.split('.')
        first_part = parts[0]
        if '_' in first_part and len(first_part) > 10:
            # This looks like a run_id, return depth of hierarchical part
            # Root is "0" -> depth 0, "0.0" -> depth 1, etc.
            return len(parts[1:]) - 1  # Subtract 1 because root is depth 0
    
    # Fallback: count all parts and subtract 1
    return len(node_id.split('.')) - 1


def load_jsonl_nodes(file_path: Path) -> List[Dict]:
    """Load nodes from JSONL file."""
    nodes = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                nodes.append(json.loads(line))
    return nodes


def extract_text_for_embedding(node: Dict) -> str:
    """Extract text content from node for embedding."""
    # Combine prompt and response for semantic analysis
    prompt = node.get('prompt', '')
    response = node.get('response', '')
    return f"{prompt} {response}".strip()


def analyze_semantic_drift(
    jsonl_file: Path,
    backend: EmbeddingBackend,
    output_csv: Optional[Path] = None,
) -> List[Dict]:
    """Analyze semantic drift in a JSONL run file.
    
    Args:
        jsonl_file: Path to JSONL file
        backend: Embedding backend instance
        output_csv: Optional path to save CSV results
    
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
    
    # Results storage
    results = []
    
    print("Computing embeddings...")
    for i, node in enumerate(nodes):
        node_id = node['node_id']
        text = extract_text_for_embedding(node)
        
        if not text:
            print(f"Warning: No text content for node {node_id}")
            continue
        
        # Compute embedding
        embedding = backend.encode(text)
        embeddings[node_id] = embedding
        
        # Calculate drift metrics
        depth = get_depth(node_id)
        parent_id = node.get('parent_id')
        
        # Drift from root (if not root)
        drift_from_root = None
        if depth > 0:  # Use depth instead of node_id comparison
            # Find root node (depth 0) from embeddings
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
            # Try direct parent_id first
            parent_embedding = embeddings.get(parent_id)
            if parent_embedding is None:
                # Try to find parent by hierarchical relationship
                # Extract run_id prefix if present
                if '.' in node_id:
                    parts = node_id.split('.')
                    if '_' in parts[0] and len(parts[0]) > 10:
                        # This has run_id prefix, construct parent ID with same prefix
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
        results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(nodes)} nodes")
    
    # Save to CSV if requested
    if output_csv:
        save_results_to_csv(results, output_csv)
        print(f"Results saved to {output_csv}")
    
    return results


def save_results_to_csv(results: List[Dict], output_path: Path):
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
        args.output = args.jsonl_file.with_name(f"{args.jsonl_file.stem}_drift.csv")
    
    # Create embedding backend
    print(f"Creating embedding backend: {args.backend}")
    backend = create_embedding_backend(args.backend, args.model)
    
    # Run analysis
    results = analyze_semantic_drift(
        args.jsonl_file, 
        backend, 
        args.output if not args.summary_only else None
    )
    
    # Print summary
    print_summary_stats(results)


if __name__ == "__main__":
    main()
