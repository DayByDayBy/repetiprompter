#!/usr/bin/env python3
"""
Visualization generator for JSONL run analysis.

Creates interactive Plotly HTML visualizations from CSV analysis outputs.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def load_drift_csv(csv_path: Path) -> pd.DataFrame:
    """Load semantic drift CSV."""
    return pd.read_csv(csv_path)


def load_stats_csv(csv_path: Path) -> pd.DataFrame:
    """Load structural stats CSV."""
    return pd.read_csv(csv_path)


def plot_drift_vs_depth(df: pd.DataFrame, output_path: Path):
    """Plot semantic drift vs depth with mean and variance."""
    
    # Group by depth
    depth_stats = df.groupby('depth').agg({
        'drift_from_root': ['mean', 'std', 'count'],
        'drift_from_parent': ['mean', 'std', 'count']
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Drift from Root vs Depth', 'Drift from Parent vs Depth'),
        vertical_spacing=0.12
    )
    
    # Drift from root
    if 'drift_from_root' in df.columns:
        root_mean = depth_stats[('drift_from_root', 'mean')]
        root_std = depth_stats[('drift_from_root', 'std')]
        depths = depth_stats['depth']
        
        fig.add_trace(
            go.Scatter(
                x=depths,
                y=root_mean,
                mode='lines+markers',
                name='Mean drift from root',
                line=dict(color='blue', width=2),
                marker=dict(size=8),
                error_y=dict(
                    type='data',
                    array=root_std,
                    visible=True
                )
            ),
            row=1, col=1
        )
    
    # Drift from parent
    if 'drift_from_parent' in df.columns:
        parent_mean = depth_stats[('drift_from_parent', 'mean')]
        parent_std = depth_stats[('drift_from_parent', 'std')]
        
        fig.add_trace(
            go.Scatter(
                x=depths,
                y=parent_mean,
                mode='lines+markers',
                name='Mean drift from parent',
                line=dict(color='red', width=2),
                marker=dict(size=8),
                error_y=dict(
                    type='data',
                    array=parent_std,
                    visible=True
                )
            ),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Depth", row=1, col=1)
    fig.update_xaxes(title_text="Depth", row=2, col=1)
    fig.update_yaxes(title_text="Cosine Distance", row=1, col=1)
    fig.update_yaxes(title_text="Cosine Distance", row=2, col=1)
    
    fig.update_layout(
        height=800,
        title_text="Semantic Drift Analysis",
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.write_html(output_path)
    print(f"Saved drift vs depth plot to {output_path}")


def plot_drift_scatter(df: pd.DataFrame, output_path: Path):
    """Plot drift as scatter with step progression."""
    
    fig = go.Figure()
    
    # Drift from root
    if 'drift_from_root' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['step'],
                y=df['drift_from_root'],
                mode='markers',
                name='Drift from root',
                marker=dict(
                    size=8,
                    color=df['depth'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Depth")
                ),
                text=[f"Node: {nid}<br>Depth: {d}" 
                      for nid, d in zip(df['node_id'], df['depth'])],
                hovertemplate='<b>%{text}</b><br>Step: %{x}<br>Drift: %{y:.4f}<extra></extra>'
            )
        )
    
    fig.update_layout(
        title="Semantic Drift Over Generation Order",
        xaxis_title="Generation Step",
        yaxis_title="Cosine Distance from Root",
        height=600,
        hovermode='closest'
    )
    
    fig.write_html(output_path)
    print(f"Saved drift scatter plot to {output_path}")


def plot_structural_metrics(df: pd.DataFrame, output_path: Path):
    """Plot structural metrics vs depth."""
    
    # Group by depth
    depth_stats = df.groupby('depth').agg({
        'response_words': ['mean', 'std'],
        'vocab_diversity': ['mean', 'std'],
        'repetition_score': ['mean', 'std']
    }).reset_index()
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Response Length (words) vs Depth',
            'Vocabulary Diversity vs Depth',
            'Repetition Score vs Depth'
        ),
        vertical_spacing=0.1
    )
    
    depths = depth_stats['depth']
    
    # Response length
    fig.add_trace(
        go.Scatter(
            x=depths,
            y=depth_stats[('response_words', 'mean')],
            mode='lines+markers',
            name='Response length',
            line=dict(color='green', width=2),
            marker=dict(size=8),
            error_y=dict(
                type='data',
                array=depth_stats[('response_words', 'std')],
                visible=True
            )
        ),
        row=1, col=1
    )
    
    # Vocabulary diversity
    fig.add_trace(
        go.Scatter(
            x=depths,
            y=depth_stats[('vocab_diversity', 'mean')],
            mode='lines+markers',
            name='Vocab diversity',
            line=dict(color='purple', width=2),
            marker=dict(size=8),
            error_y=dict(
                type='data',
                array=depth_stats[('vocab_diversity', 'std')],
                visible=True
            )
        ),
        row=2, col=1
    )
    
    # Repetition score
    fig.add_trace(
        go.Scatter(
            x=depths,
            y=depth_stats[('repetition_score', 'mean')],
            mode='lines+markers',
            name='Repetition',
            line=dict(color='orange', width=2),
            marker=dict(size=8),
            error_y=dict(
                type='data',
                array=depth_stats[('repetition_score', 'std')],
                visible=True
            )
        ),
        row=3, col=1
    )
    
    fig.update_xaxes(title_text="Depth", row=1, col=1)
    fig.update_xaxes(title_text="Depth", row=2, col=1)
    fig.update_xaxes(title_text="Depth", row=3, col=1)
    
    fig.update_yaxes(title_text="Words", row=1, col=1)
    fig.update_yaxes(title_text="Diversity (0-1)", row=2, col=1)
    fig.update_yaxes(title_text="Repetition (0-1)", row=3, col=1)
    
    fig.update_layout(
        height=1000,
        title_text="Structural Metrics Analysis",
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.write_html(output_path)
    print(f"Saved structural metrics plot to {output_path}")


def plot_combined_overview(drift_df: pd.DataFrame, stats_df: pd.DataFrame, output_path: Path):
    """Create combined overview dashboard."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Drift from Root vs Depth',
            'Response Length vs Depth',
            'Vocabulary Diversity vs Depth',
            'Drift vs Response Length'
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # 1. Drift from root vs depth
    if 'drift_from_root' in drift_df.columns:
        depth_drift = drift_df.groupby('depth')['drift_from_root'].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=depth_drift['depth'],
                y=depth_drift['drift_from_root'],
                mode='lines+markers',
                name='Drift from root',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
    
    # 2. Response length vs depth
    depth_length = stats_df.groupby('depth')['response_words'].mean().reset_index()
    fig.add_trace(
        go.Scatter(
            x=depth_length['depth'],
            y=depth_length['response_words'],
            mode='lines+markers',
            name='Response length',
            line=dict(color='green', width=2)
        ),
        row=1, col=2
    )
    
    # 3. Vocabulary diversity vs depth
    depth_diversity = stats_df.groupby('depth')['vocab_diversity'].mean().reset_index()
    fig.add_trace(
        go.Scatter(
            x=depth_diversity['depth'],
            y=depth_diversity['vocab_diversity'],
            mode='lines+markers',
            name='Vocab diversity',
            line=dict(color='purple', width=2)
        ),
        row=2, col=1
    )
    
    # 4. Drift vs response length (correlation)
    merged = pd.merge(drift_df, stats_df, on='node_id', suffixes=('_drift', '_stats'))
    if 'drift_from_root' in merged.columns and 'response_words' in merged.columns:
        fig.add_trace(
            go.Scatter(
                x=merged['response_words'],
                y=merged['drift_from_root'],
                mode='markers',
                name='Drift vs length',
                marker=dict(
                    size=6,
                    color=merged['depth_drift'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Depth", x=1.15)
                )
            ),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text="Depth", row=1, col=1)
    fig.update_xaxes(title_text="Depth", row=1, col=2)
    fig.update_xaxes(title_text="Depth", row=2, col=1)
    fig.update_xaxes(title_text="Response Words", row=2, col=2)
    
    fig.update_yaxes(title_text="Drift", row=1, col=1)
    fig.update_yaxes(title_text="Words", row=1, col=2)
    fig.update_yaxes(title_text="Diversity", row=2, col=1)
    fig.update_yaxes(title_text="Drift", row=2, col=2)
    
    fig.update_layout(
        height=800,
        title_text="Analysis Overview Dashboard",
        showlegend=False,
        hovermode='closest'
    )
    
    fig.write_html(output_path)
    print(f"Saved combined overview to {output_path}")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Generate visualizations from analysis CSVs")
    parser.add_argument("--drift", type=Path, help="Path to drift CSV file")
    parser.add_argument("--stats", type=Path, help="Path to stats CSV file")
    parser.add_argument("--output-dir", type=Path, help="Output directory for HTML files")
    parser.add_argument("--prefix", default="viz", help="Prefix for output filenames")
    
    args = parser.parse_args()
    
    if not args.drift and not args.stats:
        print("Error: Must provide at least --drift or --stats CSV file")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Use directory of first provided CSV
        output_dir = (args.drift or args.stats).parent
    
    # Load data
    drift_df = None
    stats_df = None
    
    if args.drift:
        if not args.drift.exists():
            print(f"Error: Drift CSV {args.drift} does not exist")
            sys.exit(1)
        drift_df = load_drift_csv(args.drift)
        print(f"Loaded drift data: {len(drift_df)} nodes")
    
    if args.stats:
        if not args.stats.exists():
            print(f"Error: Stats CSV {args.stats} does not exist")
            sys.exit(1)
        stats_df = load_stats_csv(args.stats)
        print(f"Loaded stats data: {len(stats_df)} nodes")
    
    # Generate plots
    if drift_df is not None:
        plot_drift_vs_depth(drift_df, output_dir / f"{args.prefix}_drift_vs_depth.html")
        plot_drift_scatter(drift_df, output_dir / f"{args.prefix}_drift_scatter.html")
    
    if stats_df is not None:
        plot_structural_metrics(stats_df, output_dir / f"{args.prefix}_structural_metrics.html")
    
    if drift_df is not None and stats_df is not None:
        plot_combined_overview(drift_df, stats_df, output_dir / f"{args.prefix}_overview.html")
    
    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
