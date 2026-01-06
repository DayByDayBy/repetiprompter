#!/usr/bin/env python3
"""
Structural statistics analyzer for JSONL runs.

Computes surface-level metrics for each response:
- Length (chars, words, lines)
- Repetition (n-gram overlap)
- Vocabulary diversity (type/token ratio)
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def tokenize_words(text: str) -> List[str]:
    """Simple word tokenization using regex."""
    # Split on whitespace and punctuation, lowercase
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words


def count_chars(text: str) -> int:
    """Count characters in text."""
    return len(text)


def count_words(text: str) -> int:
    """Count words in text."""
    return len(tokenize_words(text))


def count_lines(text: str) -> int:
    """Count lines in text."""
    return len(text.split('\n'))


def get_ngrams(words: List[str], n: int) -> List[tuple]:
    """Extract n-grams from word list."""
    if len(words) < n:
        return []
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def repetition_score(text: str, n: int = 3) -> float:
    """Calculate repetition score as fraction of repeated n-grams.
    
    Returns fraction of n-grams that appear more than once.
    """
    words = tokenize_words(text)
    ngrams = get_ngrams(words, n)
    
    if not ngrams:
        return 0.0
    
    counts = Counter(ngrams)
    repeated = sum(1 for count in counts.values() if count > 1)
    
    return repeated / len(counts) if counts else 0.0


def vocab_diversity(text: str) -> float:
    """Calculate vocabulary diversity (type/token ratio).
    
    Returns unique_words / total_words.
    """
    words = tokenize_words(text)
    
    if not words:
        return 0.0
    
    unique_words = len(set(words))
    return unique_words / len(words)


def sentence_count(text: str) -> int:
    """Count sentences using simple heuristics."""
    # Split on sentence-ending punctuation
    sentences = re.split(r'[.!?]+', text)
    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)


def avg_sentence_length(text: str) -> float:
    """Calculate average sentence length in words."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.0
    
    total_words = sum(count_words(s) for s in sentences)
    return total_words / len(sentences)


def load_jsonl_nodes(file_path: Path) -> List[Dict]:
    """Load nodes from JSONL file."""
    nodes = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    nodes.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line {line_num} in {file_path}: {e}")
                    continue
    return nodes


def get_depth(node_id: str) -> int:
    """Get depth from node ID."""
    if '.' in node_id:
        parts = node_id.split('.')
        first_part = parts[0]
        if '_' in first_part and len(first_part) > 10:
            return len(parts[1:]) - 1
    return len(node_id.split('.')) - 1


def analyze_structural_stats(
    jsonl_file: Path,
    output_csv: Optional[Path] = None,
) -> List[Dict]:
    """Analyze structural statistics in a JSONL run file.
    
    Args:
        jsonl_file: Path to JSONL file
        output_csv: Optional path to save CSV results
    
    Returns:
        List of structural stats results
    """
    print(f"Loading nodes from {jsonl_file}...")
    nodes = load_jsonl_nodes(jsonl_file)
    print(f"Loaded {len(nodes)} nodes")
    
    results = []
    
    print("Computing structural statistics...")
    for i, node in enumerate(nodes):
        node_id = node.get('node_id')
        if not node_id:
            print(f"Warning: Node {i} missing node_id, skipping")
            continue
        
        response = node.get('response', '')
        prompt = node.get('prompt', '')
        
        # Compute metrics on response
        result = {
            'node_id': node_id,
            'depth': get_depth(node_id),
            'step': i,
            # Length metrics
            'response_chars': count_chars(response),
            'response_words': count_words(response),
            'response_lines': count_lines(response),
            'response_sentences': sentence_count(response),
            # Structural metrics
            'repetition_score': repetition_score(response),
            'vocab_diversity': vocab_diversity(response),
            'avg_sentence_length': avg_sentence_length(response),
            # Prompt metrics for comparison
            'prompt_chars': count_chars(prompt),
            'prompt_words': count_words(prompt),
            # Token counts from original data if available
            'prompt_tokens': node.get('prompt_eval_count'),
            'response_tokens': node.get('eval_count'),
        }
        results.append(result)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(nodes)} nodes")
    
    # Save to CSV if requested
    if output_csv:
        save_results_to_csv(results, output_csv)
        print(f"Results saved to {output_csv}")
    
    return results


def save_results_to_csv(results: List[Dict], output_path: Path):
    """Save structural stats results to CSV."""
    if not results:
        return
    
    fieldnames = list(results[0].keys())
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary_stats(results: List[Dict]):
    """Print summary statistics."""
    if not results:
        print("No results to analyze")
        return
    
    print("\n=== Structural Statistics Summary ===")
    print(f"Total nodes analyzed: {len(results)}")
    print(f"Max depth: {max(r['depth'] for r in results)}")
    
    # Response length stats
    chars = [r['response_chars'] for r in results]
    words = [r['response_words'] for r in results]
    
    print(f"\nResponse length (chars):")
    print(f"  Mean: {np.mean(chars):.1f}")
    print(f"  Std:  {np.std(chars):.1f}")
    print(f"  Min:  {np.min(chars)}")
    print(f"  Max:  {np.max(chars)}")
    
    print(f"\nResponse length (words):")
    print(f"  Mean: {np.mean(words):.1f}")
    print(f"  Std:  {np.std(words):.1f}")
    print(f"  Min:  {np.min(words)}")
    print(f"  Max:  {np.max(words)}")
    
    # Repetition and diversity
    rep_scores = [r['repetition_score'] for r in results]
    diversity = [r['vocab_diversity'] for r in results]
    
    print(f"\nRepetition score (3-gram):")
    print(f"  Mean: {np.mean(rep_scores):.4f}")
    print(f"  Std:  {np.std(rep_scores):.4f}")
    
    print(f"\nVocabulary diversity:")
    print(f"  Mean: {np.mean(diversity):.4f}")
    print(f"  Std:  {np.std(diversity):.4f}")
    
    # Group by depth
    depth_stats = {}
    for result in results:
        depth = result['depth']
        if depth not in depth_stats:
            depth_stats[depth] = {'words': [], 'diversity': [], 'repetition': []}
        depth_stats[depth]['words'].append(result['response_words'])
        depth_stats[depth]['diversity'].append(result['vocab_diversity'])
        depth_stats[depth]['repetition'].append(result['repetition_score'])
    
    print(f"\nMetrics by depth:")
    for depth in sorted(depth_stats.keys()):
        stats = depth_stats[depth]
        print(f"  Depth {depth}: words={np.mean(stats['words']):.1f}, "
              f"diversity={np.mean(stats['diversity']):.3f}, "
              f"repetition={np.mean(stats['repetition']):.3f}, "
              f"n={len(stats['words'])}")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Analyze structural stats in JSONL runs")
    parser.add_argument("jsonl_file", type=Path, help="Path to JSONL run file")
    parser.add_argument(
        "--output", 
        type=Path, 
        help="Output CSV file (default: <jsonl_file>_stats.csv)"
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
        args.output = args.jsonl_file.with_name(f"{args.jsonl_file.stem}_stats.csv")
    
    # Run analysis
    results = analyze_structural_stats(
        args.jsonl_file, 
        args.output if not args.summary_only else None
    )
    
    # Print summary
    print_summary_stats(results)


if __name__ == "__main__":
    main()
