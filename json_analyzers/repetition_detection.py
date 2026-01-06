#!/usr/bin/env python3
"""
Repetition detection analyzer for JSONL runs.

Three-layer detection:
1. Lexical: N-gram reuse, exact span repetition, self-overlap
2. Structural: Template reuse, discourse markers, shape similarity
3. Semantic: Embedding similarity, novelty score, loop detection
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from .core.tree_utils import (
    get_depth,
    get_ancestors,
    load_jsonl_nodes,
    build_node_index,
)
from .core.text_utils import (
    tokenize_words,
    tokenize_sentences,
    get_ngrams,
    extract_discourse_markers,
    extract_sentence_templates,
    word_overlap,
)

try:
    from modular_rep_set.embedding_backend import (
        cosine_similarity,
        create_embedding_backend,
        EmbeddingBackend,
    )
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False


# ============================================================================
# LAYER 1: Lexical Repetition
# ============================================================================

def ngram_reuse_ratio(text: str, n: int) -> float:
    """Calculate fraction of n-grams that appear more than once.
    
    Args:
        text: Input text
        n: N-gram size
        
    Returns:
        Ratio of repeated n-grams [0, 1]
    """
    words = tokenize_words(text)
    ngrams = get_ngrams(words, n)
    
    if not ngrams:
        return 0.0
    
    counts = Counter(ngrams)
    repeated = sum(1 for count in counts.values() if count > 1)
    
    return repeated / len(counts) if counts else 0.0


def find_repeated_spans(text: str, min_length: int = 10) -> List[Tuple[int, int, str]]:
    """Find repeated substrings of at least min_length characters.
    
    Uses naive O(n²) approach. For very long texts, consider suffix arrays.
    
    Args:
        text: Input text
        min_length: Minimum span length
        
    Returns:
        List of (start, end, substring) tuples
    """
    text = text.lower()
    n = len(text)
    spans = []
    
    if n < min_length * 2:
        return spans
    
    for length in range(min_length, min(100, n // 2) + 1):
        seen = {}
        for i in range(n - length + 1):
            substring = text[i:i+length]
            if substring in seen:
                spans.append((i, i + length, substring))
            else:
                seen[substring] = i
    
    return spans


def exact_span_ratio(text: str, min_length: int = 10) -> float:
    """Calculate fraction of text covered by repeated spans.
    
    Args:
        text: Input text
        min_length: Minimum span length
        
    Returns:
        Ratio of text in repeated spans [0, 1]
    """
    if len(text) < min_length:
        return 0.0
    
    spans = find_repeated_spans(text, min_length)
    
    if not spans:
        return 0.0
    
    covered = set()
    for start, end, _ in spans:
        for i in range(start, end):
            covered.add(i)
    
    return len(covered) / len(text)


def compute_self_overlap(
    current_words: Set[str],
    prior_words: Set[str],
) -> float:
    """Compute Jaccard similarity between current and prior word sets.
    
    Args:
        current_words: Words in current response
        prior_words: Accumulated words from prior responses
        
    Returns:
        Jaccard similarity [0, 1]
    """
    if not current_words and not prior_words:
        return 0.0
    if not current_words or not prior_words:
        return 0.0
    
    intersection = len(current_words & prior_words)
    union = len(current_words | prior_words)
    
    return intersection / union if union > 0 else 0.0


def analyze_lexical_layer(
    response: str,
    prior_responses: List[str],
) -> Dict:
    """Analyze lexical repetition metrics.
    
    Args:
        response: Current response text
        prior_responses: List of prior responses in branch
        
    Returns:
        Dictionary with lexical metrics
    """
    current_words = set(tokenize_words(response))
    prior_words = set()
    for prior in prior_responses:
        prior_words.update(tokenize_words(prior))
    
    return {
        'ngram_reuse_2': ngram_reuse_ratio(response, 2),
        'ngram_reuse_3': ngram_reuse_ratio(response, 3),
        'ngram_reuse_5': ngram_reuse_ratio(response, 5),
        'exact_span_ratio': exact_span_ratio(response),
        'self_overlap': compute_self_overlap(current_words, prior_words),
    }


# ============================================================================
# LAYER 2: Structural Repetition
# ============================================================================

def compute_template_reuse(
    sentences: List[str],
    prior_templates: Set[str],
) -> Tuple[float, Set[str]]:
    """Compute fraction of sentences matching prior templates.
    
    Args:
        sentences: Sentences in current response
        prior_templates: Set of templates from prior responses
        
    Returns:
        Tuple of (reuse_ratio, updated_templates)
    """
    if not sentences:
        return 0.0, prior_templates
    
    current_templates = extract_sentence_templates(sentences)
    
    if not prior_templates:
        return 0.0, set(current_templates)
    
    matches = sum(1 for t in current_templates if t in prior_templates)
    reuse_ratio = matches / len(current_templates) if current_templates else 0.0
    
    updated = prior_templates | set(current_templates)
    
    return reuse_ratio, updated


def compute_discourse_marker_density(text: str) -> float:
    """Compute discourse markers per sentence.
    
    Args:
        text: Input text
        
    Returns:
        Density (markers per sentence)
    """
    sentences = tokenize_sentences(text)
    if not sentences:
        return 0.0
    
    markers = extract_discourse_markers(text)
    return len(markers) / len(sentences)


def compute_shape_similarity(
    current: str,
    parent: Optional[str],
) -> float:
    """Compute structural similarity between current and parent response.
    
    Compares:
    - Paragraph count
    - Sentence count
    - Average sentence length
    - List/bullet presence
    
    Args:
        current: Current response
        parent: Parent response (if any)
        
    Returns:
        Similarity score [0, 1]
    """
    if not parent:
        return 0.0
    
    def extract_features(text: str) -> np.ndarray:
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        sentences = tokenize_sentences(text)
        words = tokenize_words(text)
        
        has_bullets = 1.0 if any(line.strip().startswith(('-', '*', '•')) 
                                  for line in text.split('\n')) else 0.0
        has_numbers = 1.0 if any(line.strip()[:2].rstrip('.').isdigit() 
                                  for line in text.split('\n') if line.strip()) else 0.0
        
        avg_sent_len = len(words) / len(sentences) if sentences else 0.0
        
        return np.array([
            len(paragraphs),
            len(sentences),
            avg_sent_len / 50.0,
            has_bullets,
            has_numbers,
        ], dtype=float)
    
    feat_current = extract_features(current)
    feat_parent = extract_features(parent)
    
    norm_current = np.linalg.norm(feat_current)
    norm_parent = np.linalg.norm(feat_parent)
    
    if norm_current == 0 or norm_parent == 0:
        return 0.0
    
    similarity = np.dot(feat_current, feat_parent) / (norm_current * norm_parent)
    return float(max(0.0, similarity))


def analyze_structural_layer(
    response: str,
    parent_response: Optional[str],
    prior_templates: Set[str],
) -> Tuple[Dict, Set[str]]:
    """Analyze structural repetition metrics.
    
    Args:
        response: Current response text
        parent_response: Parent response (if any)
        prior_templates: Templates from prior responses
        
    Returns:
        Tuple of (metrics dict, updated templates)
    """
    sentences = tokenize_sentences(response)
    template_reuse, updated_templates = compute_template_reuse(sentences, prior_templates)
    
    return {
        'template_reuse': template_reuse,
        'discourse_marker_density': compute_discourse_marker_density(response),
        'shape_similarity': compute_shape_similarity(response, parent_response),
    }, updated_templates


# ============================================================================
# LAYER 3: Semantic Repetition
# ============================================================================

def analyze_semantic_layer(
    response: str,
    prior_responses: List[str],
    prior_embeddings: List[np.ndarray],
    backend: Optional['EmbeddingBackend'],
    loop_threshold: float = 0.95,
) -> Tuple[Dict, Optional[np.ndarray]]:
    """Analyze semantic repetition metrics.
    
    Args:
        response: Current response text
        prior_responses: Prior responses in branch
        prior_embeddings: Embeddings of prior responses
        backend: Embedding backend (optional)
        loop_threshold: Similarity threshold for loop detection
        
    Returns:
        Tuple of (metrics dict, current embedding or None)
    """
    if not backend or not HAS_EMBEDDINGS:
        return {
            'semantic_self_sim': None,
            'novelty_score': None,
            'info_per_token': None,
            'loop_detected': False,
        }, None
    
    try:
        current_embedding = backend.encode(response)
    except Exception:
        return {
            'semantic_self_sim': None,
            'novelty_score': None,
            'info_per_token': None,
            'loop_detected': False,
        }, None
    
    semantic_self_sim = None
    if prior_embeddings:
        semantic_self_sim = cosine_similarity(current_embedding, prior_embeddings[-1])
    
    novelty_score = 1.0
    loop_detected = False
    
    if prior_embeddings:
        max_sim = 0.0
        for prior_emb in prior_embeddings:
            sim = cosine_similarity(current_embedding, prior_emb)
            max_sim = max(max_sim, sim)
            if sim >= loop_threshold:
                loop_detected = True
        novelty_score = 1.0 - max_sim
    
    words = tokenize_words(response)
    unique_words = len(set(words))
    vocab_diversity = unique_words / len(words) if words else 0.0
    info_per_token = novelty_score * vocab_diversity if words else 0.0
    
    return {
        'semantic_self_sim': semantic_self_sim,
        'novelty_score': novelty_score,
        'info_per_token': info_per_token,
        'loop_detected': loop_detected,
    }, current_embedding


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_repetition_detection(
    jsonl_file: Path,
    output_csv: Optional[Path] = None,
    use_embeddings: bool = True,
    embedding_backend: str = "ollama",
    embedding_model: Optional[str] = None,
    loop_threshold: float = 0.95,
) -> List[Dict]:
    """Analyze repetition detection metrics for a JSONL run file.
    
    Args:
        jsonl_file: Path to JSONL file
        output_csv: Optional path to save CSV results
        use_embeddings: Whether to use embedding-based metrics
        embedding_backend: Backend for embeddings
        embedding_model: Model for embeddings
        loop_threshold: Threshold for loop detection
    
    Returns:
        List of repetition analysis results
    """
    print(f"Loading nodes from {jsonl_file}...")
    nodes = load_jsonl_nodes(jsonl_file)
    print(f"Loaded {len(nodes)} nodes")
    
    node_index = build_node_index(nodes)
    
    backend = None
    if use_embeddings and HAS_EMBEDDINGS:
        try:
            print(f"Creating embedding backend: {embedding_backend}")
            backend = create_embedding_backend(embedding_backend, embedding_model)
        except Exception as e:
            print(f"Warning: Could not create embedding backend: {e}")
            backend = None
    
    results = []
    prior_templates: Set[str] = set()
    embedding_cache: Dict[str, np.ndarray] = {}
    
    print("Analyzing repetition metrics...")
    for i, node in enumerate(nodes):
        node_id = node.get('node_id', '')
        response = node.get('response', '')
        parent_id = node.get('parent_id')
        
        ancestors = get_ancestors(node_id)
        prior_responses = []
        prior_embeddings = []
        
        for ancestor_id in ancestors:
            if ancestor_id in node_index:
                ancestor_response = node_index[ancestor_id].get('response', '')
                prior_responses.append(ancestor_response)
                if ancestor_id in embedding_cache:
                    prior_embeddings.append(embedding_cache[ancestor_id])
        
        parent_response = None
        if parent_id and parent_id in node_index:
            parent_response = node_index[parent_id].get('response', '')
        
        lexical = analyze_lexical_layer(response, prior_responses)
        
        structural, prior_templates = analyze_structural_layer(
            response, parent_response, prior_templates
        )
        
        semantic, current_embedding = analyze_semantic_layer(
            response, prior_responses, prior_embeddings, backend, loop_threshold
        )
        
        if current_embedding is not None:
            embedding_cache[node_id] = current_embedding
        
        result = {
            'node_id': node_id,
            'depth': get_depth(node_id),
            'step': i,
            **lexical,
            **structural,
            **semantic,
        }
        results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(nodes)} nodes")
    
    if output_csv:
        save_results_to_csv(results, output_csv)
        print(f"Results saved to {output_csv}")
    
    return results


def save_results_to_csv(results: List[Dict], output_path: Path):
    """Save repetition analysis results to CSV."""
    if not results:
        return
    
    fieldnames = [
        'node_id', 'depth', 'step',
        'ngram_reuse_2', 'ngram_reuse_3', 'ngram_reuse_5',
        'exact_span_ratio', 'self_overlap',
        'template_reuse', 'discourse_marker_density', 'shape_similarity',
        'semantic_self_sim', 'novelty_score', 'info_per_token', 'loop_detected',
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary_stats(results: List[Dict]):
    """Print summary statistics."""
    if not results:
        print("No results to analyze")
        return
    
    print("\n=== Repetition Detection Summary ===")
    print(f"Total nodes analyzed: {len(results)}")
    
    ngram3 = [r['ngram_reuse_3'] for r in results]
    print(f"\n3-gram reuse:")
    print(f"  Mean: {np.mean(ngram3):.4f}")
    print(f"  Max:  {np.max(ngram3):.4f}")
    
    novelty = [r['novelty_score'] for r in results if r['novelty_score'] is not None]
    if novelty:
        print(f"\nNovelty score:")
        print(f"  Mean: {np.mean(novelty):.4f}")
        print(f"  Min:  {np.min(novelty):.4f}")
    
    loops = sum(1 for r in results if r['loop_detected'])
    print(f"\nLoops detected: {loops}")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Analyze repetition detection in JSONL runs")
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
        "--loop-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for loop detection"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file (default: <jsonl_file>_repetition.csv)"
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
        args.output = args.jsonl_file.with_name(f"{args.jsonl_file.stem}_repetition.csv")
    
    results = analyze_repetition_detection(
        args.jsonl_file,
        args.output if not args.summary_only else None,
        use_embeddings=not args.no_embeddings,
        embedding_backend=args.backend,
        embedding_model=args.model,
        loop_threshold=args.loop_threshold,
    )
    
    print_summary_stats(results)


if __name__ == "__main__":
    main()
