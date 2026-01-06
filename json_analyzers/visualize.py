#!/usr/bin/env python3
"""
Visualization generator for JSONL run analysis.

Creates interactive Plotly HTML visualizations from CSV analysis outputs.

Supported plot types:
- Drift: drift_vs_depth, drift_scatter, drift_extended
- Structural: structural_metrics
- Repetition: repetition_heatmap, loop_timeline
- Quality: quality_radar, quality_trends
- Convergence: convergence_plateau
- Combined: overview dashboard
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


def plot_repetition_heatmap(df: pd.DataFrame, output_path: Path):
    """Plot repetition metrics as heatmap by step."""
    
    metrics = ['ngram_reuse_2', 'ngram_reuse_3', 'ngram_reuse_5', 
               'exact_span_ratio', 'self_overlap', 'template_reuse']
    available = [m for m in metrics if m in df.columns]
    
    if not available:
        print("No repetition metrics found in CSV")
        return
    
    # Create matrix
    matrix = df[available].values.T
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=df['step'],
        y=available,
        colorscale='Reds',
        hoverongaps=False,
        hovertemplate='Step: %{x}<br>Metric: %{y}<br>Value: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Repetition Metrics Heatmap',
        xaxis_title='Generation Step',
        yaxis_title='Metric',
        height=400
    )
    
    fig.write_html(output_path)
    print(f"Saved repetition heatmap to {output_path}")


def plot_loop_timeline(df: pd.DataFrame, output_path: Path):
    """Plot loop detection timeline with novelty scores."""
    
    if 'novelty_score' not in df.columns:
        print("No novelty_score found in CSV")
        return
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Novelty Score Over Time', 'Loop Detection Events'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Novelty score line
    fig.add_trace(
        go.Scatter(
            x=df['step'],
            y=df['novelty_score'],
            mode='lines+markers',
            name='Novelty',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Loop detection markers
    if 'loop_detected' in df.columns:
        loops = df[df['loop_detected'] == True]
        fig.add_trace(
            go.Scatter(
                x=loops['step'],
                y=[1] * len(loops),
                mode='markers',
                name='Loop detected',
                marker=dict(color='red', size=15, symbol='x')
            ),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text='Step', row=1, col=1)
    fig.update_xaxes(title_text='Step', row=2, col=1)
    fig.update_yaxes(title_text='Novelty (0-1)', row=1, col=1)
    fig.update_yaxes(title_text='', showticklabels=False, row=2, col=1)
    
    fig.update_layout(
        height=600,
        title_text='Loop Detection Timeline',
        showlegend=True
    )
    
    fig.write_html(output_path)
    print(f"Saved loop timeline to {output_path}")


def plot_quality_radar(df: pd.DataFrame, output_path: Path):
    """Plot quality metrics as radar chart (averaged)."""
    
    metrics = {
        'coherence_variance': 'Coherence',
        'info_density': 'Info Density',
        'prompt_coverage': 'Prompt Coverage',
        'entity_count': 'Entity Count',
    }
    
    available = {k: v for k, v in metrics.items() if k in df.columns}
    if not available:
        print("No quality metrics found in CSV")
        return
    
    # Normalize values to 0-1 range
    values = []
    labels = []
    for col, label in available.items():
        val = df[col].mean()
        max_val = df[col].max()
        if max_val > 0:
            val = val / max_val
        values.append(val)
        labels.append(label)
    
    # Close the radar
    values.append(values[0])
    labels.append(labels[0])
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Quality Profile'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title='Quality Metrics Radar (Normalized)',
        height=500
    )
    
    fig.write_html(output_path)
    print(f"Saved quality radar to {output_path}")


def plot_quality_trends(df: pd.DataFrame, output_path: Path):
    """Plot quality metrics over generation steps."""
    
    metrics = ['hedge_density', 'prompt_coverage', 'info_density', 'redundancy_score']
    available = [m for m in metrics if m in df.columns]
    
    if not available:
        print("No quality trend metrics found in CSV")
        return
    
    fig = make_subplots(
        rows=len(available), cols=1,
        subplot_titles=[m.replace('_', ' ').title() for m in available],
        vertical_spacing=0.08
    )
    
    colors = ['blue', 'green', 'purple', 'orange']
    
    for i, metric in enumerate(available, 1):
        fig.add_trace(
            go.Scatter(
                x=df['step'],
                y=df[metric],
                mode='lines+markers',
                name=metric,
                line=dict(color=colors[i-1], width=2),
                marker=dict(size=5)
            ),
            row=i, col=1
        )
        fig.update_xaxes(title_text='Step', row=i, col=1)
    
    fig.update_layout(
        height=200 * len(available) + 100,
        title_text='Quality Metrics Over Time',
        showlegend=False
    )
    
    fig.write_html(output_path)
    print(f"Saved quality trends to {output_path}")


def plot_convergence_plateau(df: pd.DataFrame, output_path: Path):
    """Plot convergence analysis with plateau highlighting."""
    
    if 'drift_from_root' not in df.columns:
        print("No drift_from_root found in convergence CSV")
        return
    
    fig = go.Figure()
    
    # Main drift line
    fig.add_trace(
        go.Scatter(
            x=df['step'],
            y=df['drift_from_root'],
            mode='lines+markers',
            name='Drift from root',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        )
    )
    
    # Highlight plateau regions
    if 'in_plateau' in df.columns:
        plateau_df = df[df['in_plateau'] == True]
        if len(plateau_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=plateau_df['step'],
                    y=plateau_df['drift_from_root'],
                    mode='markers',
                    name='In plateau',
                    marker=dict(color='yellow', size=12, symbol='square',
                                line=dict(color='orange', width=2))
                )
            )
    
    # Mark fixed point
    if 'is_fixed_point' in df.columns:
        fixed = df[df['is_fixed_point'] == True]
        if len(fixed) > 0:
            fig.add_trace(
                go.Scatter(
                    x=fixed['step'],
                    y=fixed['drift_from_root'],
                    mode='markers',
                    name='Fixed point',
                    marker=dict(color='red', size=15, symbol='star')
                )
            )
    
    # Gradient subplot
    if 'drift_gradient' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['step'],
                y=df['drift_gradient'],
                mode='lines',
                name='Gradient',
                line=dict(color='gray', width=1, dash='dot'),
                yaxis='y2'
            )
        )
    
    fig.update_layout(
        title='Convergence Analysis',
        xaxis_title='Generation Step',
        yaxis_title='Drift from Root',
        yaxis2=dict(
            title='Gradient',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        height=500,
        hovermode='x unified'
    )
    
    fig.write_html(output_path)
    print(f"Saved convergence plateau plot to {output_path}")


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
    parser.add_argument("--repetition", type=Path, help="Path to repetition CSV file")
    parser.add_argument("--quality", type=Path, help="Path to quality CSV file")
    parser.add_argument("--convergence", type=Path, help="Path to convergence CSV file")
    parser.add_argument("--output-dir", type=Path, help="Output directory for HTML files")
    parser.add_argument("--prefix", default="viz", help="Prefix for output filenames")
    
    args = parser.parse_args()
    
    all_csvs = [args.drift, args.stats, args.repetition, args.quality, args.convergence]
    if not any(all_csvs):
        print("Error: Must provide at least one CSV file")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Use directory of first provided CSV
        for csv_path in all_csvs:
            if csv_path:
                output_dir = csv_path.parent
                break
    
    # Load and plot drift data
    drift_df = None
    if args.drift:
        if not args.drift.exists():
            print(f"Error: Drift CSV {args.drift} does not exist")
            sys.exit(1)
        drift_df = load_drift_csv(args.drift)
        print(f"Loaded drift data: {len(drift_df)} nodes")
        plot_drift_vs_depth(drift_df, output_dir / f"{args.prefix}_drift_vs_depth.html")
        plot_drift_scatter(drift_df, output_dir / f"{args.prefix}_drift_scatter.html")
    
    # Load and plot stats data
    stats_df = None
    if args.stats:
        if not args.stats.exists():
            print(f"Error: Stats CSV {args.stats} does not exist")
            sys.exit(1)
        stats_df = load_stats_csv(args.stats)
        print(f"Loaded stats data: {len(stats_df)} nodes")
        plot_structural_metrics(stats_df, output_dir / f"{args.prefix}_structural_metrics.html")
    
    # Load and plot repetition data
    if args.repetition:
        if not args.repetition.exists():
            print(f"Error: Repetition CSV {args.repetition} does not exist")
            sys.exit(1)
        rep_df = pd.read_csv(args.repetition)
        print(f"Loaded repetition data: {len(rep_df)} nodes")
        plot_repetition_heatmap(rep_df, output_dir / f"{args.prefix}_repetition_heatmap.html")
        plot_loop_timeline(rep_df, output_dir / f"{args.prefix}_loop_timeline.html")
    
    # Load and plot quality data
    if args.quality:
        if not args.quality.exists():
            print(f"Error: Quality CSV {args.quality} does not exist")
            sys.exit(1)
        qual_df = pd.read_csv(args.quality)
        print(f"Loaded quality data: {len(qual_df)} nodes")
        plot_quality_radar(qual_df, output_dir / f"{args.prefix}_quality_radar.html")
        plot_quality_trends(qual_df, output_dir / f"{args.prefix}_quality_trends.html")
    
    # Load and plot convergence data
    if args.convergence:
        if not args.convergence.exists():
            print(f"Error: Convergence CSV {args.convergence} does not exist")
            sys.exit(1)
        conv_df = pd.read_csv(args.convergence)
        print(f"Loaded convergence data: {len(conv_df)} nodes")
        plot_convergence_plateau(conv_df, output_dir / f"{args.prefix}_convergence.html")
    
    # Combined overview (if drift and stats both available)
    if drift_df is not None and stats_df is not None:
        plot_combined_overview(drift_df, stats_df, output_dir / f"{args.prefix}_overview.html")
    
    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
