#!/usr/bin/env python3
"""
Orchestrator for running all JSONL analyzers.

Runs all analyzers on a JSONL file and generates a combined report.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


ANALYZERS = {
    'structural': {
        'module': 'json_analyzers.structural_stats',
        'function': 'analyze_structural_stats',
        'suffix': '_stats.csv',
        'requires_embeddings': False,
    },
    'drift': {
        'module': 'json_analyzers.semantic_drift',
        'function': 'analyze_semantic_drift',
        'suffix': '_drift_ext.csv',
        'requires_embeddings': True,
    },
    'repetition': {
        'module': 'json_analyzers.repetition_detection',
        'function': 'analyze_repetition_detection',
        'suffix': '_repetition.csv',
        'requires_embeddings': True,
    },
    'quality': {
        'module': 'json_analyzers.quality_signals',
        'function': 'analyze_quality_signals',
        'suffix': '_quality.csv',
        'requires_embeddings': True,
    },
    'convergence': {
        'module': 'json_analyzers.convergence_tracker',
        'function': 'analyze_convergence',
        'suffix': '_convergence.csv',
        'requires_embeddings': True,
    },
}


def run_analyzer(
    name: str,
    jsonl_file: Path,
    output_dir: Path,
    backend: str = "ollama",
    model: Optional[str] = None,
    skip_embeddings: bool = False,
) -> Optional[Path]:
    """Run a single analyzer and return the output CSV path.
    
    Args:
        name: Analyzer name
        jsonl_file: Input JSONL file
        output_dir: Output directory for CSV
        backend: Embedding backend
        model: Embedding model
        skip_embeddings: Skip embedding-based analyzers
        
    Returns:
        Path to output CSV or None if skipped/failed
    """
    config = ANALYZERS.get(name)
    if not config:
        print(f"Unknown analyzer: {name}")
        return None
    
    if skip_embeddings and config['requires_embeddings']:
        print(f"Skipping {name} (requires embeddings)")
        return None
    
    output_csv = output_dir / f"{jsonl_file.stem}{config['suffix']}"
    
    print(f"\n{'='*60}")
    print(f"Running {name} analyzer...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        module = __import__(config['module'], fromlist=[config['function']])
        func = getattr(module, config['function'])
        
        if name == 'structural':
            func(jsonl_file, output_csv)
        elif name == 'drift':
            from modular_rep_set.embedding_backend import create_embedding_backend
            emb_backend = create_embedding_backend(backend, model)
            func(jsonl_file, emb_backend, output_csv)
        elif name == 'repetition':
            func(jsonl_file, output_csv, use_embeddings=not skip_embeddings,
                 embedding_backend=backend, embedding_model=model)
        elif name == 'quality':
            func(jsonl_file, output_csv, use_embeddings=not skip_embeddings,
                 embedding_backend=backend, embedding_model=model)
        elif name == 'convergence':
            func(jsonl_file, output_csv, use_embeddings=not skip_embeddings,
                 embedding_backend=backend, embedding_model=model)
        
        elapsed = time.time() - start_time
        print(f"Completed {name} in {elapsed:.1f}s -> {output_csv}")
        return output_csv
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Error running {name} after {elapsed:.1f}s: {e}")
        return None


def run_visualizations(
    output_dir: Path,
    csv_files: Dict[str, Path],
    prefix: str = "viz",
):
    """Generate visualizations from CSV files.
    
    Args:
        output_dir: Output directory
        csv_files: Dictionary of analyzer name -> CSV path
        prefix: Prefix for output filenames
    """
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}")
    
    try:
        from json_analyzers.visualize import (
            load_drift_csv, load_stats_csv,
            plot_drift_vs_depth, plot_drift_scatter,
            plot_structural_metrics,
            plot_repetition_heatmap, plot_loop_timeline,
            plot_quality_radar, plot_quality_trends,
            plot_convergence_plateau,
            plot_combined_overview,
        )
        import pandas as pd
        
        drift_df = None
        stats_df = None
        
        if 'drift' in csv_files:
            drift_df = load_drift_csv(csv_files['drift'])
            plot_drift_vs_depth(drift_df, output_dir / f"{prefix}_drift_vs_depth.html")
            plot_drift_scatter(drift_df, output_dir / f"{prefix}_drift_scatter.html")
        
        if 'structural' in csv_files:
            stats_df = load_stats_csv(csv_files['structural'])
            plot_structural_metrics(stats_df, output_dir / f"{prefix}_structural_metrics.html")
        
        if 'repetition' in csv_files:
            rep_df = pd.read_csv(csv_files['repetition'])
            plot_repetition_heatmap(rep_df, output_dir / f"{prefix}_repetition_heatmap.html")
            plot_loop_timeline(rep_df, output_dir / f"{prefix}_loop_timeline.html")
        
        if 'quality' in csv_files:
            qual_df = pd.read_csv(csv_files['quality'])
            plot_quality_radar(qual_df, output_dir / f"{prefix}_quality_radar.html")
            plot_quality_trends(qual_df, output_dir / f"{prefix}_quality_trends.html")
        
        if 'convergence' in csv_files:
            conv_df = pd.read_csv(csv_files['convergence'])
            plot_convergence_plateau(conv_df, output_dir / f"{prefix}_convergence.html")
        
        if drift_df is not None and stats_df is not None:
            plot_combined_overview(drift_df, stats_df, output_dir / f"{prefix}_overview.html")
        
        print("Visualizations complete!")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")


def generate_report(
    jsonl_file: Path,
    output_dir: Path,
    csv_files: Dict[str, Path],
    elapsed_times: Dict[str, float],
):
    """Generate a summary report.
    
    Args:
        jsonl_file: Input JSONL file
        output_dir: Output directory
        csv_files: Dictionary of analyzer name -> CSV path
        elapsed_times: Timing information
    """
    report_path = output_dir / f"{jsonl_file.stem}_report.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# Analysis Report: {jsonl_file.name}\n\n")
        f.write(f"**Input file:** `{jsonl_file}`\n\n")
        f.write(f"**Output directory:** `{output_dir}`\n\n")
        
        f.write("## Analyzers Run\n\n")
        f.write("| Analyzer | Status | Output |\n")
        f.write("|----------|--------|--------|\n")
        
        for name in ANALYZERS:
            if name in csv_files:
                f.write(f"| {name} | ✅ Complete | `{csv_files[name].name}` |\n")
            else:
                f.write(f"| {name} | ⏭️ Skipped | - |\n")
        
        f.write("\n## Output Files\n\n")
        f.write("### CSV Files\n")
        for name, path in csv_files.items():
            f.write(f"- `{path.name}` ({name} analysis)\n")
        
        f.write("\n### Visualizations\n")
        html_files = list(output_dir.glob("viz_*.html"))
        for html_file in sorted(html_files):
            f.write(f"- `{html_file.name}`\n")
        
        f.write("\n## Timing\n\n")
        total = sum(elapsed_times.values())
        f.write(f"**Total time:** {total:.1f}s\n\n")
    
    print(f"Report saved to {report_path}")


def run_all(
    jsonl_file: Path,
    output_dir: Optional[Path] = None,
    analyzers: Optional[List[str]] = None,
    backend: str = "ollama",
    model: Optional[str] = None,
    skip_embeddings: bool = False,
    skip_visualizations: bool = False,
):
    """Run all analyzers on a JSONL file.
    
    Args:
        jsonl_file: Input JSONL file
        output_dir: Output directory (default: same as input)
        analyzers: List of analyzers to run (default: all)
        backend: Embedding backend
        model: Embedding model
        skip_embeddings: Skip embedding-based analyzers
        skip_visualizations: Skip visualization generation
    """
    if not jsonl_file.exists():
        print(f"Error: File {jsonl_file} does not exist")
        sys.exit(1)
    
    if output_dir is None:
        output_dir = jsonl_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if analyzers is None:
        analyzers = list(ANALYZERS.keys())
    
    print(f"Running analyzers on {jsonl_file}")
    print(f"Output directory: {output_dir}")
    print(f"Analyzers: {', '.join(analyzers)}")
    print(f"Embedding backend: {backend}")
    if skip_embeddings:
        print("Note: Embedding-based analyzers will be skipped")
    
    csv_files: Dict[str, Path] = {}
    elapsed_times: Dict[str, float] = {}
    
    total_start = time.time()
    
    for name in analyzers:
        start = time.time()
        result = run_analyzer(name, jsonl_file, output_dir, backend, model, skip_embeddings)
        elapsed = time.time() - start
        elapsed_times[name] = elapsed
        
        if result:
            csv_files[name] = result
    
    if not skip_visualizations and csv_files:
        run_visualizations(output_dir, csv_files)
    
    generate_report(jsonl_file, output_dir, csv_files, elapsed_times)
    
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Output directory: {output_dir}")
    print(f"CSV files: {len(csv_files)}")
    print(f"Report: {jsonl_file.stem}_report.md")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run all JSONL analyzers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s runs/my_run.jsonl
  %(prog)s runs/my_run.jsonl --analyzers structural drift
  %(prog)s runs/my_run.jsonl --skip-embeddings
  %(prog)s runs/my_run.jsonl --backend sentence-transformers
        """
    )
    parser.add_argument("jsonl_file", type=Path, help="Path to JSONL run file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: same as input file)"
    )
    parser.add_argument(
        "--analyzers",
        nargs="+",
        choices=list(ANALYZERS.keys()),
        help="Analyzers to run (default: all)"
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "sentence-transformers", "huggingface"],
        default="ollama",
        help="Embedding backend"
    )
    parser.add_argument("--model", help="Model name for embedding backend")
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip analyzers that require embeddings"
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip visualization generation"
    )
    
    args = parser.parse_args()
    
    run_all(
        args.jsonl_file,
        args.output_dir,
        args.analyzers,
        args.backend,
        args.model,
        args.skip_embeddings,
        args.skip_visualizations,
    )


if __name__ == "__main__":
    main()
