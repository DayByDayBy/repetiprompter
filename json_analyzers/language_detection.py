#!/usr/bin/env python3
"""
Language detection analyzer for JSONL runs.

Per-segment language identification with entropy tracking:
- Sentence-level language detection
- Language distribution entropy
- Language switch detection
- Code block exclusion
"""

import argparse
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from .core.tree_utils import get_depth, load_jsonl_nodes
from .core.text_utils import tokenize_sentences, tokenize_words, extract_code_blocks

try:
    from langdetect import detect, detect_langs, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

try:
    from lingua import Language, LanguageDetectorBuilder
    HAS_LINGUA = True
except ImportError:
    HAS_LINGUA = False


MIN_SENTENCE_LENGTH = 20


class LanguageDetector:
    """Language detection wrapper with fallback support."""
    
    def __init__(self, backend: str = "auto"):
        """Initialize language detector.
        
        Args:
            backend: "lingua", "langdetect", or "auto" (tries lingua first)
        """
        self.backend = backend
        self._lingua_detector = None
        
        if backend == "auto":
            if HAS_LINGUA:
                self.backend = "lingua"
            elif HAS_LANGDETECT:
                self.backend = "langdetect"
            else:
                raise ImportError(
                    "No language detection library available. "
                    "Install with: uv add langdetect  OR  uv add lingua-language-detector"
                )
        
        if self.backend == "lingua":
            if not HAS_LINGUA:
                raise ImportError("lingua not installed. Install with: uv add lingua-language-detector")
            self._lingua_detector = LanguageDetectorBuilder.from_all_languages().build()
        elif self.backend == "langdetect":
            if not HAS_LANGDETECT:
                raise ImportError("langdetect not installed. Install with: uv add langdetect")
    
    def detect(self, text: str) -> Tuple[str, float]:
        """Detect language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if len(text.strip()) < MIN_SENTENCE_LENGTH:
            return ("unk", 0.0)
        
        try:
            if self.backend == "lingua":
                result = self._lingua_detector.detect_language_of(text)
                if result is None:
                    return ("unk", 0.0)
                confidence = self._lingua_detector.compute_language_confidence(text, result)
                return (result.iso_code_639_1.name.lower(), confidence)
            else:
                langs = detect_langs(text)
                if langs:
                    return (langs[0].lang, langs[0].prob)
                return ("unk", 0.0)
        except Exception:
            return ("unk", 0.0)


def shannon_entropy(counts: Counter) -> float:
    """Calculate Shannon entropy from counts.
    
    Args:
        counts: Counter of items
        
    Returns:
        Entropy in bits
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy


def analyze_language(
    text: str,
    detector: LanguageDetector,
) -> Dict:
    """Analyze language characteristics of text.
    
    Args:
        text: Text to analyze
        detector: Language detector instance
        
    Returns:
        Dictionary with language metrics
    """
    text_no_code, code_blocks = extract_code_blocks(text)
    
    sentences = tokenize_sentences(text_no_code)
    
    if not sentences:
        return {
            'dominant_lang': 'unk',
            'lang_entropy': 0.0,
            'lang_consistency': 1.0,
            'non_primary_pct': 0.0,
            'switch_count': 0,
        }
    
    sentence_langs = []
    for sentence in sentences:
        if len(sentence.strip()) >= MIN_SENTENCE_LENGTH:
            lang, conf = detector.detect(sentence)
            sentence_langs.append(lang)
    
    if not sentence_langs:
        return {
            'dominant_lang': 'unk',
            'lang_entropy': 0.0,
            'lang_consistency': 1.0,
            'non_primary_pct': 0.0,
            'switch_count': 0,
        }
    
    lang_counts = Counter(sentence_langs)
    dominant_lang = lang_counts.most_common(1)[0][0]
    
    entropy = shannon_entropy(lang_counts)
    
    num_unique = len(lang_counts)
    if num_unique > 1:
        max_entropy = math.log2(num_unique)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        normalized_entropy = 0.0
    
    lang_consistency = 1.0 - normalized_entropy
    
    non_primary_count = sum(1 for lang in sentence_langs if lang != dominant_lang)
    non_primary_pct = (non_primary_count / len(sentence_langs)) * 100 if sentence_langs else 0.0
    
    switch_count = 0
    for i in range(1, len(sentence_langs)):
        if sentence_langs[i] != sentence_langs[i-1]:
            switch_count += 1
    
    return {
        'dominant_lang': dominant_lang,
        'lang_entropy': entropy,
        'lang_consistency': lang_consistency,
        'non_primary_pct': non_primary_pct,
        'switch_count': switch_count,
    }


def analyze_language_detection(
    jsonl_file: Path,
    output_csv: Optional[Path] = None,
    backend: str = "auto",
) -> List[Dict]:
    """Analyze language detection metrics for a JSONL run file.
    
    Args:
        jsonl_file: Path to JSONL file
        output_csv: Optional path to save CSV results
        backend: Language detection backend ("auto", "lingua", "langdetect")
    
    Returns:
        List of language analysis results
    """
    print(f"Loading nodes from {jsonl_file}...")
    nodes = load_jsonl_nodes(jsonl_file)
    print(f"Loaded {len(nodes)} nodes")
    
    print(f"Initializing language detector (backend: {backend})...")
    detector = LanguageDetector(backend)
    print(f"Using backend: {detector.backend}")
    
    results = []
    
    print("Analyzing language metrics...")
    for i, node in enumerate(nodes):
        node_id = node.get('node_id', '')
        response = node.get('response', '')
        
        lang_metrics = analyze_language(response, detector)
        
        result = {
            'node_id': node_id,
            'depth': get_depth(node_id),
            'step': i,
            **lang_metrics,
        }
        results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(nodes)} nodes")
    
    if output_csv:
        save_results_to_csv(results, output_csv)
        print(f"Results saved to {output_csv}")
    
    return results


def save_results_to_csv(results: List[Dict], output_path: Path):
    """Save language analysis results to CSV."""
    if not results:
        return
    
    fieldnames = [
        'node_id', 'depth', 'step',
        'dominant_lang', 'lang_entropy', 'lang_consistency',
        'non_primary_pct', 'switch_count',
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
    
    print("\n=== Language Detection Summary ===")
    print(f"Total nodes analyzed: {len(results)}")
    
    lang_counts = Counter(r['dominant_lang'] for r in results)
    print(f"\nLanguage distribution:")
    for lang, count in lang_counts.most_common():
        pct = (count / len(results)) * 100
        print(f"  {lang}: {count} ({pct:.1f}%)")
    
    consistencies = [r['lang_consistency'] for r in results]
    print(f"\nLanguage consistency:")
    print(f"  Mean: {sum(consistencies)/len(consistencies):.3f}")
    
    total_switches = sum(r['switch_count'] for r in results)
    print(f"\nTotal language switches: {total_switches}")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Analyze language detection in JSONL runs")
    parser.add_argument("jsonl_file", type=Path, help="Path to JSONL run file")
    parser.add_argument(
        "--backend",
        choices=["auto", "lingua", "langdetect"],
        default="auto",
        help="Language detection backend"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file (default: <jsonl_file>_lang.csv)"
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
        args.output = args.jsonl_file.with_name(f"{args.jsonl_file.stem}_lang.csv")
    
    results = analyze_language_detection(
        args.jsonl_file,
        args.output if not args.summary_only else None,
        args.backend,
    )
    
    print_summary_stats(results)


if __name__ == "__main__":
    main()
