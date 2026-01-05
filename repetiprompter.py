#!/usr/bin/env python3
"""
Repetiprompter - LLM degradation through recursive self-prompting.

Usage:
    python repetiprompter.py run --config config.yaml
    python repetiprompter.py run --config config.yaml --override model.temperature=0.9
"""

from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table

from modular_rep_set.config_loader import load_config, apply_overrides, parse_override
from modular_rep_set.runner import Runner
from modular_rep_set.framing_strategies import list_strategies
from modular_rep_set.ollama_interface import OllamaClient

app = typer.Typer(
    name="repetiprompter",
    help="LLM degradation through recursive self-prompting",
    add_completion=False
)
console = Console()


@app.command()
def run(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to YAML configuration file",
        exists=True,
        readable=True
    ),
    override: Optional[List[str]] = typer.Option(
        None,
        "--override", "-o",
        help="Override config values (e.g., model.temperature=0.9)"
    )
):
    """Run a generation experiment from a config file."""
    
    console.print(f"[bold]Loading config:[/bold] {config}")
    
    try:
        run_config = load_config(config)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)
    
    if override:
        overrides = {}
        for o in override:
            try:
                key, value = parse_override(o)
                overrides[key] = value
            except ValueError as e:
                console.print(f"[red]Invalid override:[/red] {e}")
                raise typer.Exit(1)
        
        run_config = apply_overrides(run_config, overrides)
        console.print(f"[dim]Applied {len(overrides)} override(s)[/dim]")
    
    console.print()
    _show_run_summary(run_config)
    console.print()
    
    try:
        runner = Runner(run_config)
        output_path = runner.run()
        console.print()
        console.print(f"[green bold]✓ Complete![/green bold] Output: {output_path}")
    except Exception as e:
        console.print(f"[red]Error during run:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def strategies():
    """List available framing strategies."""
    table = Table(title="Framing Strategies")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    
    for s in list_strategies():
        table.add_row(s["name"], s["description"])
    
    console.print(table)


@app.command()
def models():
    """List available Ollama models."""
    client = OllamaClient()
    model_list = client.list_models()
    
    if not model_list:
        console.print("[yellow]No models found. Is Ollama running?[/yellow]")
        return
    
    table = Table(title="Available Models")
    table.add_column("Model Name", style="cyan")
    
    for name in sorted(model_list):
        table.add_row(name)
    
    console.print(table)
    console.print(f"\n[dim]{len(model_list)} model(s) available[/dim]")


def _show_run_summary(config):
    """Display a summary of the run configuration."""
    table = Table(title="Run Configuration", show_header=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    
    table.add_row("Run ID", config.run_identity.run_id)
    table.add_row("Model", config.model.model_name)
    table.add_row("Mode", config.topology.mode.value)
    
    if config.topology.mode.value == "chain":
        table.add_row("Steps", str(config.topology.chain.steps))
    else:
        table.add_row("Depth", str(config.topology.tree.depth))
        table.add_row("Branching", str(config.topology.tree.branching_factor))
    
    table.add_row("Framing", config.prompting.framing_strategy.value)
    table.add_row("Temp Regime", config.temperature_regime.type.value)
    table.add_row("Reminder", "enabled" if config.reminder.enabled else "disabled")
    table.add_row("Output Dir", config.output.output_dir)
    
    prompt_preview = config.prompting.initial_prompt[:60]
    if len(config.prompting.initial_prompt) > 60:
        prompt_preview += "..."
    table.add_row("Initial Prompt", prompt_preview)
    
    console.print(table)


if __name__ == "__main__":
    app()
