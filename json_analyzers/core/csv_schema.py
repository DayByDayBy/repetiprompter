"""
CSV schema definitions and validation for json_analyzers.

Provides:
- Schema definitions for all analyzer outputs
- CSV validation functions
- CSV merging utilities
- Documentation generation
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


@dataclass
class ColumnDef:
    """Definition for a CSV column."""
    name: str
    dtype: str  # 'str', 'int', 'float', 'bool'
    description: str
    nullable: bool = True
    
    def validate(self, value: Any) -> bool:
        """Check if value matches column type."""
        if value is None or (isinstance(value, str) and value == ''):
            return self.nullable
        
        try:
            if self.dtype == 'str':
                return isinstance(value, str)
            elif self.dtype == 'int':
                int(value)
                return True
            elif self.dtype == 'float':
                float(value)
                return True
            elif self.dtype == 'bool':
                return isinstance(value, bool) or value in ('True', 'False', '0', '1', 0, 1)
            return True
        except (ValueError, TypeError):
            return False


@dataclass
class CSVSchema:
    """Schema definition for a CSV output."""
    name: str
    description: str
    columns: List[ColumnDef]
    merge_key: str = 'node_id'
    
    def column_names(self) -> List[str]:
        """Get list of column names."""
        return [col.name for col in self.columns]
    
    def validate_row(self, row: Dict[str, Any]) -> List[str]:
        """Validate a row against schema. Returns list of errors."""
        errors = []
        for col in self.columns:
            if col.name not in row:
                if not col.nullable:
                    errors.append(f"Missing required column: {col.name}")
            elif not col.validate(row[col.name]):
                errors.append(f"Invalid value for {col.name}: {row[col.name]}")
        return errors
    
    def to_markdown(self) -> str:
        """Generate markdown documentation for schema."""
        lines = [
            f"## {self.name}",
            "",
            self.description,
            "",
            "| Column | Type | Nullable | Description |",
            "|--------|------|----------|-------------|",
        ]
        for col in self.columns:
            nullable = "Yes" if col.nullable else "No"
            lines.append(f"| `{col.name}` | {col.dtype} | {nullable} | {col.description} |")
        return '\n'.join(lines)


# Common columns used across schemas
NODE_ID_COL = ColumnDef('node_id', 'str', 'Unique node identifier', nullable=False)
DEPTH_COL = ColumnDef('depth', 'int', 'Tree depth (0 = root)', nullable=False)
STEP_COL = ColumnDef('step', 'int', 'Generation order', nullable=False)


# Schema definitions for each analyzer

DRIFT_SCHEMA = CSVSchema(
    name="Semantic Drift",
    description="Semantic drift metrics from semantic_drift.py",
    columns=[
        NODE_ID_COL,
        DEPTH_COL,
        STEP_COL,
        ColumnDef('drift_from_root', 'float', 'Cosine distance to root node [0, 2]'),
        ColumnDef('drift_from_parent', 'float', 'Cosine distance to parent node [0, 2]'),
        ColumnDef('prompt_length', 'int', 'Prompt character count'),
        ColumnDef('response_length', 'int', 'Response character count'),
    ]
)

DRIFT_EXTENDED_SCHEMA = CSVSchema(
    name="Semantic Drift Extended",
    description="Extended drift metrics with local/global/instruction modes",
    columns=[
        NODE_ID_COL,
        DEPTH_COL,
        STEP_COL,
        ColumnDef('drift_from_root', 'float', 'Cosine distance to root node [0, 2]'),
        ColumnDef('drift_from_parent', 'float', 'Cosine distance to parent node [0, 2]'),
        ColumnDef('drift_local', 'float', 'Distance to previous response [0, 2]'),
        ColumnDef('drift_global', 'float', 'Distance to run centroid [0, 2]'),
        ColumnDef('drift_instruction', 'float', 'Distance to original prompt [0, 2]'),
        ColumnDef('embedding_mode', 'str', '{prompt_only, response_only, prompt_response}'),
    ]
)

STRUCTURAL_SCHEMA = CSVSchema(
    name="Structural Stats",
    description="Surface-level structural metrics from structural_stats.py",
    columns=[
        NODE_ID_COL,
        DEPTH_COL,
        STEP_COL,
        ColumnDef('response_chars', 'int', 'Response character count'),
        ColumnDef('response_words', 'int', 'Response word count'),
        ColumnDef('response_lines', 'int', 'Response line count'),
        ColumnDef('response_sentences', 'int', 'Response sentence count'),
        ColumnDef('repetition_score', 'float', 'Fraction of repeated 3-grams [0, 1]'),
        ColumnDef('vocab_diversity', 'float', 'Type/token ratio [0, 1]'),
        ColumnDef('avg_sentence_length', 'float', 'Average words per sentence'),
        ColumnDef('prompt_chars', 'int', 'Prompt character count'),
        ColumnDef('prompt_words', 'int', 'Prompt word count'),
        ColumnDef('prompt_tokens', 'int', 'Prompt token count (if available)'),
        ColumnDef('response_tokens', 'int', 'Response token count (if available)'),
    ]
)

LANGUAGE_SCHEMA = CSVSchema(
    name="Language Detection",
    description="Per-segment language identification and entropy",
    columns=[
        NODE_ID_COL,
        DEPTH_COL,
        STEP_COL,
        ColumnDef('dominant_lang', 'str', 'ISO 639-1 code of most frequent language'),
        ColumnDef('lang_entropy', 'float', 'Shannon entropy over language distribution [0, log(n)]'),
        ColumnDef('lang_consistency', 'float', '1 - normalized entropy [0, 1]'),
        ColumnDef('non_primary_pct', 'float', '% tokens not in dominant language [0, 100]'),
        ColumnDef('switch_count', 'int', 'Language switch events within response'),
    ]
)

REPETITION_SCHEMA = CSVSchema(
    name="Repetition Detection",
    description="Three-layer repetition detection (lexical, structural, semantic)",
    columns=[
        NODE_ID_COL,
        DEPTH_COL,
        STEP_COL,
        # Lexical layer
        ColumnDef('ngram_reuse_2', 'float', '2-gram reuse ratio [0, 1]'),
        ColumnDef('ngram_reuse_3', 'float', '3-gram reuse ratio [0, 1]'),
        ColumnDef('ngram_reuse_5', 'float', '5-gram reuse ratio [0, 1]'),
        ColumnDef('exact_span_ratio', 'float', 'Ratio of response in exact repeated spans ≥10 chars'),
        ColumnDef('self_overlap', 'float', 'Jaccard similarity to accumulated prior words'),
        # Structural layer
        ColumnDef('template_reuse', 'float', 'Sentence template reuse ratio [0, 1]'),
        ColumnDef('discourse_marker_density', 'float', 'Discourse markers per sentence'),
        ColumnDef('shape_similarity', 'float', 'Paragraph structure similarity to parent [0, 1]'),
        # Semantic layer
        ColumnDef('semantic_self_sim', 'float', 'Embedding similarity to previous response [0, 1]'),
        ColumnDef('novelty_score', 'float', '1 - max similarity to any prior response [0, 1]'),
        ColumnDef('info_per_token', 'float', 'Novel information bits per token (derived)'),
        ColumnDef('loop_detected', 'bool', 'True if response near-identical to ancestor'),
    ]
)

QUALITY_SCHEMA = CSVSchema(
    name="Quality Signals",
    description="Response quality metrics (coherence, specificity, alignment)",
    columns=[
        NODE_ID_COL,
        DEPTH_COL,
        STEP_COL,
        # Coherence
        ColumnDef('coherence_variance', 'float', 'Intra-response embedding variance'),
        ColumnDef('topic_shift_count', 'int', 'Detected topic shifts within response'),
        # Specificity
        ColumnDef('abstract_concrete_ratio', 'float', 'Abstract term count / concrete term count'),
        ColumnDef('hedge_density', 'float', 'Hedge phrases per 100 words'),
        ColumnDef('entity_count', 'int', 'Named entity count'),
        # Alignment
        ColumnDef('prompt_similarity', 'float', 'Cosine similarity to immediate prompt [0, 1]'),
        ColumnDef('prompt_coverage', 'float', '% of prompt terms appearing in response [0, 100]'),
        # Compression
        ColumnDef('redundancy_score', 'float', '1 - compression ratio proxy [0, 1]'),
        ColumnDef('info_density', 'float', 'Unique concepts per token'),
    ]
)

CONVERGENCE_SCHEMA = CSVSchema(
    name="Convergence Tracking",
    description="Trajectory stability and attractor detection",
    columns=[
        NODE_ID_COL,
        DEPTH_COL,
        STEP_COL,
        ColumnDef('semantic_radius', 'float', 'Distance from seed prompt [0, 2]'),
        ColumnDef('drift_velocity', 'float', 'Δ(semantic_radius) from parent'),
        ColumnDef('convergence_direction', 'str', '{converging, diverging, stable, cycling}'),
        ColumnDef('attractor_period', 'int', 'Cycle length if cycling detected, else null'),
        ColumnDef('trajectory_stability', 'float', 'Rolling variance of drift_velocity'),
    ]
)


ALL_SCHEMAS: Dict[str, CSVSchema] = {
    'drift': DRIFT_SCHEMA,
    'drift_ext': DRIFT_EXTENDED_SCHEMA,
    'structural': STRUCTURAL_SCHEMA,
    'language': LANGUAGE_SCHEMA,
    'repetition': REPETITION_SCHEMA,
    'quality': QUALITY_SCHEMA,
    'convergence': CONVERGENCE_SCHEMA,
}


def validate_csv(file_path: Path, schema: CSVSchema) -> List[str]:
    """Validate a CSV file against a schema.
    
    Args:
        file_path: Path to CSV file
        schema: Schema to validate against
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if reader.fieldnames is None:
                return ["CSV file has no headers"]
            
            schema_cols = set(schema.column_names())
            file_cols = set(reader.fieldnames)
            
            missing = schema_cols - file_cols
            extra = file_cols - schema_cols
            
            for col in missing:
                col_def = next(c for c in schema.columns if c.name == col)
                if not col_def.nullable:
                    errors.append(f"Missing required column: {col}")
            
            for row_num, row in enumerate(reader, start=2):
                row_errors = schema.validate_row(row)
                for err in row_errors:
                    errors.append(f"Row {row_num}: {err}")
                    
    except Exception as e:
        errors.append(f"Error reading CSV: {e}")
    
    return errors


def merge_csvs(
    file_paths: List[Path],
    output_path: Optional[Path] = None,
    on: str = 'node_id',
):
    """Merge multiple CSV files on a common key.
    
    Args:
        file_paths: List of CSV file paths
        output_path: Optional path to save merged CSV
        on: Column to merge on (default: 'node_id')
        
    Returns:
        Merged DataFrame (requires pandas)
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for merge_csvs. Install with: uv add pandas")
    
    if not file_paths:
        return pd.DataFrame()
    
    dfs = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    merged = dfs[0]
    for df in dfs[1:]:
        common_cols = set(merged.columns) & set(df.columns)
        suffix_cols = common_cols - {on, 'depth', 'step'}
        
        merged = pd.merge(
            merged,
            df,
            on=on,
            how='outer',
            suffixes=('', '_dup')
        )
        
        dup_cols = [c for c in merged.columns if c.endswith('_dup')]
        merged = merged.drop(columns=dup_cols)
    
    if output_path:
        merged.to_csv(output_path, index=False)
    
    return merged


def generate_schema_docs(output_path: Optional[Path] = None) -> str:
    """Generate markdown documentation for all schemas.
    
    Args:
        output_path: Optional path to save documentation
        
    Returns:
        Markdown string
    """
    lines = [
        "# CSV Output Schemas",
        "",
        "This document describes the CSV output format for each analyzer.",
        "",
    ]
    
    for name, schema in ALL_SCHEMAS.items():
        lines.append(schema.to_markdown())
        lines.append("")
    
    doc = '\n'.join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(doc)
    
    return doc
