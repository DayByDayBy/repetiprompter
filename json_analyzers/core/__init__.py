"""
Core utilities for json_analyzers package.

Provides shared functionality:
- tree_utils: Node indexing, tree traversal, file loading
- text_utils: Tokenization, sentence splitting, n-gram extraction
- csv_schema: CSV column definitions and validation
"""

from .tree_utils import (
    parse_node_id,
    get_depth,
    get_parent_id,
    get_ancestors,
    get_sibling_index,
    load_jsonl_nodes,
    load_legacy_json_tree,
    sort_by_generation_order,
    build_node_index,
)

from .text_utils import (
    tokenize_words,
    tokenize_sentences,
    get_ngrams,
    get_char_ngrams,
    extract_discourse_markers,
    extract_sentence_templates,
    is_hedge_phrase,
    count_hedge_phrases,
    normalize_whitespace,
    extract_code_blocks,
    word_overlap,
)

from .csv_schema import (
    ColumnDef,
    CSVSchema,
    ALL_SCHEMAS,
    validate_csv,
    merge_csvs,
    generate_schema_docs,
)

__all__ = [
    # tree_utils
    "parse_node_id",
    "get_depth",
    "get_parent_id",
    "get_ancestors",
    "get_sibling_index",
    "load_jsonl_nodes",
    "load_legacy_json_tree",
    "sort_by_generation_order",
    "build_node_index",
    # text_utils
    "tokenize_words",
    "tokenize_sentences",
    "get_ngrams",
    "get_char_ngrams",
    "extract_discourse_markers",
    "extract_sentence_templates",
    "is_hedge_phrase",
    "count_hedge_phrases",
    "normalize_whitespace",
    "extract_code_blocks",
    "word_overlap",
    # csv_schema
    "ColumnDef",
    "CSVSchema",
    "ALL_SCHEMAS",
    "validate_csv",
    "merge_csvs",
    "generate_schema_docs",
]
