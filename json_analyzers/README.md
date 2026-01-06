# JSON Analyzers

Advanced behavioral analysis toolkit for JSONL run files from LLM repetition experiments.

## Overview

This module provides a comprehensive suite of analyzers for detecting and measuring various behavioral patterns in LLM outputs:

- **Structural Statistics**: Basic metrics (length, vocabulary, repetition)
- **Semantic Drift**: Embedding-based drift detection with extended metrics
- **Language Detection**: Per-segment language identification and entropy
- **Repetition Detection**: Three-layer repetition analysis (lexical, structural, semantic)
- **Quality Signals**: Response quality metrics (coherence, specificity, alignment, compression)
- **Convergence Tracking**: Plateau and fixed-point detection

## Quick Start

### Run All Analyzers

```bash
uv run python3 -m json_analyzers.run_all runs/my_run.jsonl
```

This will:
1. Run all analyzers on the JSONL file
2. Generate CSV outputs for each analyzer
3. Create interactive HTML visualizations
4. Generate a summary report

### Run Individual Analyzers

```bash
# Structural statistics (no embeddings required)
uv run python3 -m json_analyzers.structural_stats runs/my_run.jsonl

# Semantic drift (requires embeddings)
uv run python3 -m json_analyzers.semantic_drift runs/my_run.jsonl --backend ollama

# Repetition detection
uv run python3 -m json_analyzers.repetition_detection runs/my_run.jsonl

# Quality signals
uv run python3 -m json_analyzers.quality_signals runs/my_run.jsonl

# Convergence tracking
uv run python3 -m json_analyzers.convergence_tracker runs/my_run.jsonl

# Language detection
uv run python3 -m json_analyzers.language_detection runs/my_run.jsonl
```

### Generate Visualizations

```bash
uv run python3 -m json_analyzers.visualize \
  --drift runs/my_run_drift_ext.csv \
  --stats runs/my_run_stats.csv \
  --repetition runs/my_run_repetition.csv \
  --quality runs/my_run_quality.csv \
  --convergence runs/my_run_convergence.csv
```

## Architecture

```
json_analyzers/
├── core/                      # Shared utilities
│   ├── tree_utils.py         # Tree traversal and node indexing
│   ├── text_utils.py         # Text processing and tokenization
│   └── csv_schema.py         # CSV schema definitions
├── structural_stats.py        # Basic structural metrics
├── semantic_drift.py          # Embedding-based drift analysis
├── language_detection.py      # Language identification
├── repetition_detection.py    # Multi-layer repetition detection
├── quality_signals.py         # Quality metrics
├── convergence_tracker.py     # Convergence analysis
├── visualize.py               # Visualization generator
└── run_all.py                 # Orchestrator
```

## Analyzer Details

### Structural Statistics

Computes basic metrics without requiring embeddings:
- Response length (chars, words, lines, sentences)
- Vocabulary diversity (unique words / total words)
- N-gram repetition scores (2-gram, 3-gram, 5-gram)
- Average sentence length

**Output**: `*_stats.csv`

### Semantic Drift

Measures semantic drift using embeddings:
- `drift_from_root`: Distance to root node
- `drift_from_parent`: Distance to parent node
- `drift_local`: Distance to previous response (step-wise)
- `drift_global`: Distance to run centroid
- `drift_instruction`: Distance to original prompt

**Embedding modes**: `prompt_only`, `response_only`, `prompt_response`

**Output**: `*_drift_ext.csv`

### Language Detection

Per-sentence language identification:
- `dominant_lang`: Most common language in response
- `lang_entropy`: Shannon entropy of language distribution
- `lang_consistency`: 1 - normalized entropy
- `non_primary_pct`: Percentage of non-dominant language
- `switch_count`: Number of language switches

**Backends**: `lingua` (recommended), `langdetect`

**Output**: `*_lang.csv`

### Repetition Detection

Three-layer analysis:

**Lexical Layer**:
- N-gram reuse ratios (2, 3, 5-grams)
- Exact span repetition
- Self-overlap with prior responses

**Structural Layer**:
- Template reuse
- Discourse marker density
- Shape similarity to parent

**Semantic Layer** (requires embeddings):
- Semantic self-similarity
- Novelty score
- Information per token
- Loop detection

**Output**: `*_repetition.csv`

### Quality Signals

Response quality metrics:

**Coherence**:
- Intra-response embedding variance
- Topic shift count

**Specificity**:
- Abstract/concrete word ratio
- Hedge phrase density
- Entity count

**Alignment**:
- Prompt similarity (embedding-based)
- Prompt coverage (word overlap)

**Compression**:
- Redundancy score (n-gram uniqueness)
- Information density (unique words / total)

**Output**: `*_quality.csv`

### Convergence Tracking

Detects when runs converge to stable attractors:
- Plateau detection via gradient analysis
- Fixed-point detection via similarity thresholds
- Drift gradient computation

**Output**: `*_convergence.csv`

## Visualization Types

The `visualize.py` module generates interactive Plotly HTML visualizations:

- **Drift plots**: drift_vs_depth, drift_scatter
- **Structural plots**: structural_metrics
- **Repetition plots**: repetition_heatmap, loop_timeline
- **Quality plots**: quality_radar, quality_trends
- **Convergence plots**: convergence_plateau
- **Combined**: overview dashboard

## Configuration

### Embedding Backends

Three backends are supported:

1. **Ollama** (default):
   ```bash
   --backend ollama --model nomic-embed-text
   ```

2. **Sentence-Transformers**:
   ```bash
   --backend sentence-transformers --model all-MiniLM-L6-v2
   ```

3. **HuggingFace**:
   ```bash
   --backend huggingface --model sentence-transformers/all-mpnet-base-v2
   ```

### Skip Embeddings

For faster analysis without embedding-based metrics:

```bash
uv run python3 -m json_analyzers.run_all runs/my_run.jsonl --skip-embeddings
```

## Output Format

All analyzers produce CSV files with a common structure:
- `node_id`: Unique node identifier
- `depth`: Tree depth (0 = root)
- `step`: Generation order index
- [analyzer-specific metrics]

CSV files can be merged on `node_id` for cross-analyzer analysis.

## Dependencies

**Required**:
- numpy
- (pandas for csv_schema merge operations)

**Optional**:
- Embedding backends: ollama, sentence-transformers, transformers
- Language detection: lingua-language-detector, langdetect
- NER: spacy
- Visualization: plotly, pandas

Install with:
```bash
uv add numpy pandas plotly sentence-transformers lingua-language-detector spacy
```

## Examples

### Full Analysis Pipeline

```bash
# Run all analyzers with embeddings
uv run python3 -m json_analyzers.run_all runs/my_run.jsonl

# Run specific analyzers
uv run python3 -m json_analyzers.run_all runs/my_run.jsonl \
  --analyzers structural drift repetition

# Use different embedding backend
uv run python3 -m json_analyzers.run_all runs/my_run.jsonl \
  --backend sentence-transformers --model all-MiniLM-L6-v2
```

### Individual Analysis

```bash
# Structural analysis only
uv run python3 -m json_analyzers.structural_stats runs/my_run.jsonl \
  --output runs/my_run_stats.csv

# Semantic drift with extended metrics
uv run python3 -m json_analyzers.semantic_drift runs/my_run.jsonl \
  --mode response_only --backend ollama

# Repetition detection without embeddings
uv run python3 -m json_analyzers.repetition_detection runs/my_run.jsonl \
  --no-embeddings
```

### Custom Visualization

```bash
# Generate specific plots
uv run python3 -m json_analyzers.visualize \
  --drift runs/my_run_drift_ext.csv \
  --stats runs/my_run_stats.csv \
  --output-dir visualizations/ \
  --prefix my_analysis
```

## Development

### Adding a New Analyzer

1. Create `json_analyzers/my_analyzer.py`
2. Implement `analyze_my_analyzer(jsonl_file, output_csv, ...)`
3. Add CSV schema to `core/csv_schema.py`
4. Add entry to `ANALYZERS` dict in `run_all.py`
5. Add visualization functions to `visualize.py`

### Core Utilities

Use shared utilities from `core/`:
- `tree_utils`: Node traversal, depth calculation, ancestor lookup
- `text_utils`: Tokenization, sentence splitting, n-grams, discourse markers
- `csv_schema`: Schema validation and documentation

## Citation

If you use this toolkit in your research, please cite:

```
@software{json_analyzers,
  title = {JSON Analyzers: Behavioral Analysis for LLM Repetition Experiments},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/[your-repo]/repetiprompter}
}
```

## License

[Your License]
