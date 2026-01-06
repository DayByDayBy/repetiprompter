"""
Tree traversal and node indexing utilities for JSONL/JSON response chains.

Supports:
- JSONL format: {node_id, parent_id, depth, prompt, response, ...} per line
- Legacy nested JSON: {content: {prompt, responses[], children[]}}

Node ID format: [run_id.]index.index.index...
- run_id (optional): timestamp-based prefix like "20260105_140916_7b71"
- indices: hierarchical position (e.g., "0.1.2" = root's child 1's child 2)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_node_id(node_id: str) -> Tuple[int, ...]:
    """Parse hierarchical node ID into tuple of integers.
    
    Handles run_id prefixes (e.g., "20260105_140916_7b71.0.1.2" -> (0, 1, 2)).
    
    Args:
        node_id: Node identifier string
        
    Returns:
        Tuple of integers representing hierarchical position
    """
    if '.' in node_id:
        parts = node_id.split('.')
        first_part = parts[0]
        if '_' in first_part and len(first_part) > 10:
            return tuple(int(part) for part in parts[1:] if part.isdigit())
    
    return tuple(int(part) for part in node_id.split('.') if part.isdigit())


def get_depth(node_id: str) -> int:
    """Get tree depth from node ID.
    
    Root node has depth 0.
    
    Args:
        node_id: Node identifier string
        
    Returns:
        Depth as integer (0 for root)
    """
    if '.' in node_id:
        parts = node_id.split('.')
        first_part = parts[0]
        if '_' in first_part and len(first_part) > 10:
            return len(parts[1:]) - 1
    
    return len(node_id.split('.')) - 1


def get_parent_id(node_id: str) -> Optional[str]:
    """Get parent node ID from a node ID.
    
    Args:
        node_id: Node identifier string
        
    Returns:
        Parent node ID, or None if root node
    """
    if '.' not in node_id:
        return None
    
    parts = node_id.split('.')
    
    if len(parts) <= 1:
        return None
    
    first_part = parts[0]
    if '_' in first_part and len(first_part) > 10:
        if len(parts) <= 2:
            return None
        return '.'.join(parts[:-1])
    
    if len(parts) == 1:
        return None
    
    return '.'.join(parts[:-1])


def get_ancestors(node_id: str) -> List[str]:
    """Get all ancestor node IDs from root to immediate parent.
    
    Args:
        node_id: Node identifier string
        
    Returns:
        List of ancestor IDs, ordered from root to parent
    """
    ancestors = []
    current = node_id
    
    while True:
        parent = get_parent_id(current)
        if parent is None:
            break
        ancestors.insert(0, parent)
        current = parent
    
    return ancestors


def get_sibling_index(node_id: str) -> int:
    """Get sibling index (position among siblings) from node ID.
    
    Args:
        node_id: Node identifier string
        
    Returns:
        Sibling index as integer
    """
    parsed = parse_node_id(node_id)
    if not parsed:
        return 0
    return parsed[-1]


def load_jsonl_nodes(file_path: Path) -> List[Dict]:
    """Load nodes from JSONL file.
    
    Each line should be a valid JSON object with at least 'node_id'.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of node dictionaries
    """
    nodes = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    nodes.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line {line_num} in {file_path}: {e}")
                    continue
    return nodes


def _flatten_legacy_tree(
    node: Dict,
    parent_id: Optional[str],
    depth: int,
    step_counter: List[int],
    result: List[Dict],
    run_id: str = "",
) -> None:
    """Recursively flatten a nested JSON tree structure.
    
    Internal helper for load_legacy_json_tree.
    """
    content = node.get('content', node)
    
    sibling_idx = step_counter[depth] if depth < len(step_counter) else 0
    while len(step_counter) <= depth:
        step_counter.append(0)
    
    if run_id:
        node_id = f"{run_id}.{'.'.join(str(s) for s in step_counter[:depth+1])}"
    else:
        node_id = '.'.join(str(s) for s in step_counter[:depth+1]) if depth > 0 else str(sibling_idx)
    
    flattened = {
        'node_id': node_id,
        'parent_id': parent_id,
        'depth': depth,
        'step': len(result),
        'prompt': content.get('prompt', ''),
        'response': '',
    }
    
    responses = content.get('responses', [])
    if responses:
        if isinstance(responses[0], str):
            flattened['response'] = responses[0]
        elif isinstance(responses[0], dict):
            flattened['response'] = responses[0].get('text', responses[0].get('content', ''))
    
    result.append(flattened)
    step_counter[depth] += 1
    
    children = content.get('children', [])
    for child in children:
        _flatten_legacy_tree(child, node_id, depth + 1, step_counter, result, run_id)


def load_legacy_json_tree(file_path: Path, run_id: str = "") -> List[Dict]:
    """Load and flatten a legacy nested JSON tree structure.
    
    Converts nested format to flat list compatible with JSONL format.
    
    Args:
        file_path: Path to JSON file
        run_id: Optional run ID prefix for node IDs
        
    Returns:
        List of flattened node dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    result: List[Dict] = []
    step_counter: List[int] = [0]
    
    if isinstance(data, list):
        for item in data:
            _flatten_legacy_tree(item, None, 0, step_counter, result, run_id)
    else:
        _flatten_legacy_tree(data, None, 0, step_counter, result, run_id)
    
    return result


def sort_by_generation_order(nodes: List[Dict]) -> List[Dict]:
    """Sort nodes by generation order (step field, then node_id).
    
    Args:
        nodes: List of node dictionaries
        
    Returns:
        Sorted list of nodes
    """
    def sort_key(node: Dict):
        if 'step' in node:
            return (node['step'], node.get('node_id', ''))
        return (parse_node_id(node.get('node_id', '')),)
    
    return sorted(nodes, key=sort_key)


def build_node_index(nodes: List[Dict]) -> Dict[str, Dict]:
    """Build lookup index from node_id to node dictionary.
    
    Enables O(1) lookup by node_id.
    
    Args:
        nodes: List of node dictionaries
        
    Returns:
        Dictionary mapping node_id to node
    """
    return {node['node_id']: node for node in nodes if 'node_id' in node}


def get_root_node_id(nodes: List[Dict]) -> Optional[str]:
    """Find the root node ID from a list of nodes.
    
    Args:
        nodes: List of node dictionaries
        
    Returns:
        Root node ID, or None if not found
    """
    for node in nodes:
        node_id = node.get('node_id', '')
        if get_depth(node_id) == 0:
            return node_id
    return None


def get_branch_nodes(node_id: str, node_index: Dict[str, Dict]) -> List[Dict]:
    """Get all nodes in the branch from root to specified node.
    
    Args:
        node_id: Target node ID
        node_index: Lookup index from build_node_index
        
    Returns:
        List of nodes from root to target, in order
    """
    ancestors = get_ancestors(node_id)
    branch = []
    
    for ancestor_id in ancestors:
        if ancestor_id in node_index:
            branch.append(node_index[ancestor_id])
    
    if node_id in node_index:
        branch.append(node_index[node_id])
    
    return branch
