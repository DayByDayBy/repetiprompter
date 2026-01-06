#!/usr/bin/env python3
"""
Convergence tracker for JSONL runs.

Detects when a run has converged to a stable attractor:
- Embedding plateau detection via gradient analysis
- Fixed-point detection via near-exact repetition
- Multi-run attractor clustering
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from .core.tree_utils import get_depth, load_jsonl_nodes, parse_node_id
from .core.text_utils import tokenize_words

try:
    from modular_rep_set.embedding_backend import (
        cosine_similarity,
        cosine_distance,
        create_embedding_backend,
        EmbeddingBackend,
    )
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False


# ============================================================================
# Gradient/Plateau Detection
# ============================================================================

def compute_gradient(values: List[float], window: int = 3) -> List[float]:
    """Compute smoothed gradient of values.
    
    Args:
        values: List of metric values
        window: Smoothing window size
        
    Returns:
        List of gradient values
    """
    if len(values) < 2:
        return [0.0] * len(values)
    
    gradients = []
    for i in range(len(values)):
        if i == 0:
            gradients.append(0.0)
        else:
            grad = values[i] - values[i-1]
            gradients.append(grad)
    
    if window > 1 and len(gradients) >= window:
        smoothed = []
        for i in range(len(gradients)):
            start = max(0, i - window // 2)
            end = min(len(gradients), i + window // 2 + 1)
            smoothed.append(np.mean(gradients[start:end]))
        return smoothed
    
    return gradients


def detect_plateau(
    values: List[float],
    threshold: float = 0.01,
    min_length: int = 3,
) -> List[Tuple[int, int]]:
    """Detect plateau regions where gradient is below threshold.
    
    Args:
        values: List of metric values
        threshold: Maximum gradient magnitude for plateau
        min_length: Minimum plateau length
        
    Returns:
        List of (start_idx, end_idx) tuples for plateau regions
    """
    if len(values) < min_length:
        return []
    
    gradients = compute_gradient(values)
    
    plateaus = []
    current_start = None
    
    for i, grad in enumerate(gradients):
        if abs(grad) < threshold:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                length = i - current_start
                if length >= min_length:
                    plateaus.append((current_start, i))
                current_start = None
    
    if current_start is not None:
        length = len(gradients) - current_start
        if length >= min_length:
            plateaus.append((current_start, len(gradients)))
    
    return plateaus


# ============================================================================
# Fixed-Point Detection
# ============================================================================

def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute word-level Jaccard similarity between texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity [0, 1]
    """
    words1 = set(tokenize_words(text1))
    words2 = set(tokenize_words(text2))
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union


def detect_fixed_point(
    responses: List[str],
    embeddings: Optional[List[np.ndarray]] = None,
    text_threshold: float = 0.95,
    embedding_threshold: float = 0.98,
) -> Optional[int]:
    """Detect first step where response becomes a fixed point.
    
    A fixed point is detected when:
    - Text similarity >= threshold with previous response, OR
    - Embedding similarity >= threshold with any prior response
    
    Args:
        responses: List of response texts
        embeddings: Optional list of embeddings
        text_threshold: Threshold for text similarity
        embedding_threshold: Threshold for embedding similarity
        
    Returns:
        Index of first fixed point, or None if not detected
    """
    if len(responses) < 2:
        return None
    
    for i in range(1, len(responses)):
        text_sim = compute_text_similarity(responses[i], responses[i-1])
        if text_sim >= text_threshold:
            return i
        
        if embeddings and len(embeddings) > i:
            for j in range(i):
                emb_sim = cosine_similarity(embeddings[i], embeddings[j])
                if emb_sim >= embedding_threshold:
                    return i
    
    return None


# ============================================================================
# Multi-Run Attractor Analysis
# ============================================================================

def cluster_final_states(
    embeddings: List[np.ndarray],
    threshold: float = 0.9,
) -> List[List[int]]:
    """Cluster final state embeddings into attractors.
    
    Simple single-linkage clustering based on similarity threshold.
    
    Args:
        embeddings: List of final state embeddings
        threshold: Similarity threshold for same cluster
        
    Returns:
        List of clusters (each cluster is a list of indices)
    """
    if not embeddings:
        return []
    
    n = len(embeddings)
    clusters: List[List[int]] = []
    assigned = set()
    
    for i in range(n):
        if i in assigned:
            continue
        
        cluster = [i]
        assigned.add(i)
        
        for j in range(i + 1, n):
            if j in assigned:
                continue
            
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                cluster.append(j)
                assigned.add(j)
        
        clusters.append(cluster)
    
    return clusters


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_convergence(
    jsonl_file: Path,
    output_csv: Optional[Path] = None,
    use_embeddings: bool = True,
    embedding_backend: str = "ollama",
    embedding_model: Optional[str] = None,
    plateau_threshold: float = 0.01,
    fixed_point_threshold: float = 0.95,
) -> Dict:
    """Analyze convergence behavior in a JSONL run file.
    
    Args:
        jsonl_file: Path to JSONL file
        output_csv: Optional path to save CSV results
        use_embeddings: Whether to use embedding-based metrics
        embedding_backend: Backend for embeddings
        embedding_model: Model for embeddings
        plateau_threshold: Gradient threshold for plateau detection
        fixed_point_threshold: Similarity threshold for fixed-point detection
    
    Returns:
        Dictionary with convergence analysis
    """
    print(f"Loading nodes from {jsonl_file}...")
    nodes = load_jsonl_nodes(jsonl_file)
    print(f"Loaded {len(nodes)} nodes")
    
    nodes.sort(key=lambda n: parse_node_id(n['node_id']))
    
    backend = None
    if use_embeddings and HAS_EMBEDDINGS:
        try:
            print(f"Creating embedding backend: {embedding_backend}")
            backend = create_embedding_backend(embedding_backend, embedding_model)
        except Exception as e:
            print(f"Warning: Could not create embedding backend: {e}")
    
    responses = [node.get('response', '') for node in nodes]
    embeddings = []
    drift_values = []
    
    print("Computing embeddings and drift values...")
    root_embedding = None
    
    for i, node in enumerate(nodes):
        response = node.get('response', '')
        
        if backend and HAS_EMBEDDINGS:
            try:
                emb = backend.encode(response)
                embeddings.append(emb)
                
                if i == 0:
                    root_embedding = emb
                    drift_values.append(0.0)
                else:
                    drift = cosine_distance(emb, root_embedding)
                    drift_values.append(drift)
            except Exception:
                pass
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(nodes)} nodes")
    
    print("Analyzing convergence patterns...")
    
    plateaus = detect_plateau(drift_values, plateau_threshold) if drift_values else []
    
    fixed_point_idx = detect_fixed_point(
        responses,
        embeddings if embeddings else None,
        text_threshold=fixed_point_threshold,
    )
    
    drift_gradient = compute_gradient(drift_values) if drift_values else []
    
    per_node_results = []
    for i, node in enumerate(nodes):
        node_id = node.get('node_id', '')
        
        in_plateau = any(start <= i < end for start, end in plateaus)
        
        result = {
            'node_id': node_id,
            'depth': get_depth(node_id),
            'step': i,
            'drift_from_root': drift_values[i] if i < len(drift_values) else None,
            'drift_gradient': drift_gradient[i] if i < len(drift_gradient) else None,
            'in_plateau': in_plateau,
            'is_fixed_point': i == fixed_point_idx if fixed_point_idx else False,
        }
        per_node_results.append(result)
    
    if output_csv:
        save_results_to_csv(per_node_results, output_csv)
        print(f"Results saved to {output_csv}")
    
    summary = {
        'total_nodes': len(nodes),
        'plateaus': plateaus,
        'fixed_point_step': fixed_point_idx,
        'converged': fixed_point_idx is not None or len(plateaus) > 0,
        'final_drift': drift_values[-1] if drift_values else None,
        'per_node_results': per_node_results,
    }
    
    return summary


def save_results_to_csv(results: List[Dict], output_path: Path):
    """Save convergence analysis results to CSV."""
    if not results:
        return
    
    fieldnames = [
        'node_id', 'depth', 'step',
        'drift_from_root', 'drift_gradient',
        'in_plateau', 'is_fixed_point',
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(summary: Dict):
    """Print convergence analysis summary."""
    print("\n=== Convergence Analysis Summary ===")
    print(f"Total nodes: {summary['total_nodes']}")
    print(f"Converged: {summary['converged']}")
    
    if summary['fixed_point_step'] is not None:
        print(f"Fixed point detected at step: {summary['fixed_point_step']}")
    
    if summary['plateaus']:
        print(f"Plateau regions: {len(summary['plateaus'])}")
        for i, (start, end) in enumerate(summary['plateaus']):
            print(f"  Plateau {i+1}: steps {start}-{end} (length: {end-start})")
    
    if summary['final_drift'] is not None:
        print(f"Final drift from root: {summary['final_drift']:.4f}")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Analyze convergence in JSONL runs")
    parser.add_argument("jsonl_file", type=Path, help="Path to JSONL run file")
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable embedding-based metrics"
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "sentence-transformers", "huggingface"],
        default="ollama",
        help="Embedding backend"
    )
    parser.add_argument("--model", help="Model name for embedding backend")
    parser.add_argument(
        "--plateau-threshold",
        type=float,
        default=0.01,
        help="Gradient threshold for plateau detection"
    )
    parser.add_argument(
        "--fixed-point-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for fixed-point detection"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file (default: <jsonl_file>_convergence.csv)"
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
    
    if not args.output and not args.summary_only:
        args.output = args.jsonl_file.with_name(f"{args.jsonl_file.stem}_convergence.csv")
    
    summary = analyze_convergence(
        args.jsonl_file,
        args.output if not args.summary_only else None,
        use_embeddings=not args.no_embeddings,
        embedding_backend=args.backend,
        embedding_model=args.model,
        plateau_threshold=args.plateau_threshold,
        fixed_point_threshold=args.fixed_point_threshold,
    )
    
    print_summary(summary)


if __name__ == "__main__":
    main()
