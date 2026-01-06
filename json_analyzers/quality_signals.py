#!/usr/bin/env python3
"""
Quality signals analyzer for JSONL runs.

Response quality metrics:
1. Coherence: Intra-response embedding variance, topic shifts
2. Specificity: Abstract/concrete ratio, hedge density, entity count
3. Alignment: Prompt similarity, prompt coverage
4. Compression: Redundancy score, info density
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from .core.tree_utils import get_depth, load_jsonl_nodes
from .core.text_utils import (
    tokenize_words,
    tokenize_sentences,
    get_ngrams,
    count_hedge_phrases,
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

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


# ============================================================================
# Word Lists for Specificity
# ============================================================================

ABSTRACT_WORDS: Set[str] = {
    "concept", "idea", "theory", "principle", "notion", "aspect", "element",
    "factor", "issue", "matter", "thing", "situation", "circumstance",
    "condition", "state", "nature", "quality", "characteristic", "property",
    "attribute", "feature", "trait", "tendency", "pattern", "approach",
    "method", "process", "system", "structure", "framework", "model",
    "relationship", "connection", "association", "correlation", "influence",
    "effect", "impact", "result", "outcome", "consequence", "implication",
    "significance", "importance", "relevance", "meaning", "sense", "context",
    "perspective", "viewpoint", "opinion", "belief", "assumption", "hypothesis",
    "possibility", "probability", "likelihood", "potential", "capability",
    "ability", "capacity", "tendency", "behavior", "action", "activity",
}

CONCRETE_WORDS: Set[str] = {
    "table", "chair", "door", "window", "wall", "floor", "ceiling", "room",
    "house", "building", "street", "road", "car", "bus", "train", "plane",
    "book", "paper", "pen", "pencil", "computer", "phone", "screen", "keyboard",
    "mouse", "file", "folder", "document", "image", "photo", "video", "audio",
    "water", "food", "drink", "coffee", "tea", "bread", "meat", "fruit",
    "tree", "flower", "grass", "leaf", "rock", "stone", "sand", "dirt",
    "sun", "moon", "star", "cloud", "rain", "snow", "wind", "fire",
    "dog", "cat", "bird", "fish", "horse", "cow", "pig", "chicken",
    "man", "woman", "child", "boy", "girl", "baby", "person", "people",
    "hand", "foot", "head", "eye", "ear", "nose", "mouth", "arm", "leg",
    "red", "blue", "green", "yellow", "black", "white", "brown", "orange",
    "one", "two", "three", "four", "five", "ten", "hundred", "thousand",
}


# ============================================================================
# Coherence Metrics
# ============================================================================

def compute_coherence_variance(
    sentences: List[str],
    backend: Optional['EmbeddingBackend'],
) -> Tuple[float, int]:
    """Compute intra-response embedding variance and topic shifts.
    
    Args:
        sentences: List of sentences
        backend: Embedding backend (optional)
        
    Returns:
        Tuple of (variance, topic_shift_count)
    """
    if not sentences or not backend or not HAS_EMBEDDINGS:
        return 0.0, 0
    
    if len(sentences) == 1:
        return 0.0, 0
    
    try:
        embeddings = backend.batch_encode(sentences)
    except Exception:
        return 0.0, 0
    
    if not embeddings:
        return 0.0, 0
    
    embeddings_array = np.array(embeddings)
    
    centroid = np.mean(embeddings_array, axis=0)
    distances = [np.linalg.norm(emb - centroid) for emb in embeddings_array]
    variance = float(np.var(distances))
    
    topic_shift_count = 0
    threshold = 0.5
    for i in range(1, len(embeddings)):
        sim = cosine_similarity(embeddings[i], embeddings[i-1])
        if sim < threshold:
            topic_shift_count += 1
    
    return variance, topic_shift_count


def analyze_coherence(
    response: str,
    backend: Optional['EmbeddingBackend'],
) -> Dict:
    """Analyze coherence metrics.
    
    Args:
        response: Response text
        backend: Embedding backend
        
    Returns:
        Dictionary with coherence metrics
    """
    sentences = tokenize_sentences(response)
    variance, shifts = compute_coherence_variance(sentences, backend)
    
    return {
        'coherence_variance': variance,
        'topic_shift_count': shifts,
    }


# ============================================================================
# Specificity Metrics
# ============================================================================

def compute_abstract_concrete_ratio(words: List[str]) -> float:
    """Compute ratio of abstract to concrete words.
    
    Args:
        words: List of words
        
    Returns:
        Ratio (abstract_count / concrete_count), or 0.0 if no concrete words
    """
    abstract_count = sum(1 for w in words if w.lower() in ABSTRACT_WORDS)
    concrete_count = sum(1 for w in words if w.lower() in CONCRETE_WORDS)
    
    if concrete_count == 0:
        return float(abstract_count) if abstract_count > 0 else 0.0
    
    return abstract_count / concrete_count


def compute_hedge_density(text: str, word_count: int) -> float:
    """Compute hedge phrases per 100 words.
    
    Args:
        text: Input text
        word_count: Total word count
        
    Returns:
        Hedge density (per 100 words)
    """
    if word_count == 0:
        return 0.0
    
    hedge_count = count_hedge_phrases(text)
    return (hedge_count / word_count) * 100


def count_entities_simple(text: str) -> int:
    """Count entities using simple heuristics (no NER).
    
    Counts capitalized multi-word sequences and quoted terms.
    
    Args:
        text: Input text
        
    Returns:
        Approximate entity count
    """
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    
    quoted = re.findall(r'"[^"]+"|\'[^\']+\'', text)
    
    return len(capitalized) + len(quoted)


def count_entities_spacy(text: str, nlp) -> int:
    """Count entities using spaCy NER.
    
    Args:
        text: Input text
        nlp: spaCy model
        
    Returns:
        Entity count
    """
    doc = nlp(text[:100000])
    return len(doc.ents)


def analyze_specificity(
    response: str,
    nlp=None,
) -> Dict:
    """Analyze specificity metrics.
    
    Args:
        response: Response text
        nlp: Optional spaCy model
        
    Returns:
        Dictionary with specificity metrics
    """
    words = tokenize_words(response)
    word_count = len(words)
    
    abstract_concrete = compute_abstract_concrete_ratio(words)
    hedge_density = compute_hedge_density(response, word_count)
    
    if nlp and HAS_SPACY:
        entity_count = count_entities_spacy(response, nlp)
    else:
        entity_count = count_entities_simple(response)
    
    return {
        'abstract_concrete_ratio': abstract_concrete,
        'hedge_density': hedge_density,
        'entity_count': entity_count,
    }


# ============================================================================
# Alignment Metrics
# ============================================================================

def compute_prompt_similarity(
    response: str,
    prompt: str,
    backend: Optional['EmbeddingBackend'],
) -> Optional[float]:
    """Compute cosine similarity between response and prompt embeddings.
    
    Args:
        response: Response text
        prompt: Prompt text
        backend: Embedding backend
        
    Returns:
        Similarity [0, 1] or None if not available
    """
    if not backend or not HAS_EMBEDDINGS or not prompt or not response:
        return None
    
    try:
        response_emb = backend.encode(response)
        prompt_emb = backend.encode(prompt)
        return cosine_similarity(response_emb, prompt_emb)
    except Exception:
        return None


def compute_prompt_coverage(response: str, prompt: str) -> float:
    """Compute percentage of prompt content words appearing in response.
    
    Args:
        response: Response text
        prompt: Prompt text
        
    Returns:
        Coverage percentage [0, 100]
    """
    if not prompt:
        return 0.0
    
    prompt_words = set(tokenize_words(prompt))
    response_words = set(tokenize_words(response))
    
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                 'from', 'as', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'between', 'under', 'again', 'further',
                 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 'just', 'and', 'but', 'or', 'if',
                 'because', 'until', 'while', 'this', 'that', 'these', 'those',
                 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
                 'who', 'whom', 'whose', 'my', 'your', 'his', 'her', 'its',
                 'our', 'their', 'me', 'him', 'us', 'them'}
    
    content_words = prompt_words - stopwords
    
    if not content_words:
        return 0.0
    
    matched = content_words & response_words
    return (len(matched) / len(content_words)) * 100


def analyze_alignment(
    response: str,
    prompt: str,
    backend: Optional['EmbeddingBackend'],
) -> Dict:
    """Analyze alignment metrics.
    
    Args:
        response: Response text
        prompt: Prompt text
        backend: Embedding backend
        
    Returns:
        Dictionary with alignment metrics
    """
    return {
        'prompt_similarity': compute_prompt_similarity(response, prompt, backend),
        'prompt_coverage': compute_prompt_coverage(response, prompt),
    }


# ============================================================================
# Compression Metrics
# ============================================================================

def compute_redundancy_score(text: str, n: int = 3) -> float:
    """Compute redundancy as 1 - (unique n-grams / total n-grams).
    
    Args:
        text: Input text
        n: N-gram size
        
    Returns:
        Redundancy score [0, 1]
    """
    words = tokenize_words(text)
    ngrams = get_ngrams(words, n)
    
    if not ngrams:
        return 0.0
    
    unique = len(set(ngrams))
    total = len(ngrams)
    
    return 1.0 - (unique / total)


def compute_info_density(words: List[str]) -> float:
    """Compute information density as unique lemmas per token.
    
    Simple version: unique words / total words (without lemmatization).
    
    Args:
        words: List of words
        
    Returns:
        Info density [0, 1]
    """
    if not words:
        return 0.0
    
    unique = len(set(words))
    return unique / len(words)


def analyze_compression(response: str) -> Dict:
    """Analyze compression metrics.
    
    Args:
        response: Response text
        
    Returns:
        Dictionary with compression metrics
    """
    words = tokenize_words(response)
    
    return {
        'redundancy_score': compute_redundancy_score(response),
        'info_density': compute_info_density(words),
    }


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_quality_signals(
    jsonl_file: Path,
    output_csv: Optional[Path] = None,
    use_embeddings: bool = True,
    embedding_backend: str = "ollama",
    embedding_model: Optional[str] = None,
    use_spacy: bool = False,
) -> List[Dict]:
    """Analyze quality signals for a JSONL run file.
    
    Args:
        jsonl_file: Path to JSONL file
        output_csv: Optional path to save CSV results
        use_embeddings: Whether to use embedding-based metrics
        embedding_backend: Backend for embeddings
        embedding_model: Model for embeddings
        use_spacy: Whether to use spaCy for NER
    
    Returns:
        List of quality analysis results
    """
    print(f"Loading nodes from {jsonl_file}...")
    nodes = load_jsonl_nodes(jsonl_file)
    print(f"Loaded {len(nodes)} nodes")
    
    backend = None
    if use_embeddings and HAS_EMBEDDINGS:
        try:
            print(f"Creating embedding backend: {embedding_backend}")
            backend = create_embedding_backend(embedding_backend, embedding_model)
        except Exception as e:
            print(f"Warning: Could not create embedding backend: {e}")
    
    nlp = None
    if use_spacy and HAS_SPACY:
        try:
            print("Loading spaCy model...")
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"Warning: Could not load spaCy: {e}")
    
    results = []
    
    print("Analyzing quality signals...")
    for i, node in enumerate(nodes):
        node_id = node.get('node_id', '')
        response = node.get('response', '')
        prompt = node.get('prompt', '')
        
        coherence = analyze_coherence(response, backend)
        specificity = analyze_specificity(response, nlp)
        alignment = analyze_alignment(response, prompt, backend)
        compression = analyze_compression(response)
        
        result = {
            'node_id': node_id,
            'depth': get_depth(node_id),
            'step': i,
            **coherence,
            **specificity,
            **alignment,
            **compression,
        }
        results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(nodes)} nodes")
    
    if output_csv:
        save_results_to_csv(results, output_csv)
        print(f"Results saved to {output_csv}")
    
    return results


def save_results_to_csv(results: List[Dict], output_path: Path):
    """Save quality analysis results to CSV."""
    if not results:
        return
    
    fieldnames = [
        'node_id', 'depth', 'step',
        'coherence_variance', 'topic_shift_count',
        'abstract_concrete_ratio', 'hedge_density', 'entity_count',
        'prompt_similarity', 'prompt_coverage',
        'redundancy_score', 'info_density',
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
    
    print("\n=== Quality Signals Summary ===")
    print(f"Total nodes analyzed: {len(results)}")
    
    coherence = [r['coherence_variance'] for r in results]
    print(f"\nCoherence variance:")
    print(f"  Mean: {np.mean(coherence):.4f}")
    
    hedge = [r['hedge_density'] for r in results]
    print(f"\nHedge density (per 100 words):")
    print(f"  Mean: {np.mean(hedge):.2f}")
    
    coverage = [r['prompt_coverage'] for r in results]
    print(f"\nPrompt coverage:")
    print(f"  Mean: {np.mean(coverage):.1f}%")
    
    info = [r['info_density'] for r in results]
    print(f"\nInfo density:")
    print(f"  Mean: {np.mean(info):.3f}")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Analyze quality signals in JSONL runs")
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
        "--spacy",
        action="store_true",
        help="Use spaCy for NER (requires en_core_web_sm)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file (default: <jsonl_file>_quality.csv)"
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
        args.output = args.jsonl_file.with_name(f"{args.jsonl_file.stem}_quality.csv")
    
    results = analyze_quality_signals(
        args.jsonl_file,
        args.output if not args.summary_only else None,
        use_embeddings=not args.no_embeddings,
        embedding_backend=args.backend,
        embedding_model=args.model,
        use_spacy=args.spacy,
    )
    
    print_summary_stats(results)


if __name__ == "__main__":
    main()
