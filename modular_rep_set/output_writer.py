"""JSONL streaming writer with stable hierarchical node IDs."""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import uuid

from .models import NodeOutput, RunConfig


class JSONLWriter:
    """
    Writes NodeOutput objects to JSONL file with streaming and crash recovery.
    
    Node IDs are hierarchical, encoding tree structure:
    - Root: "{run_id}.0"
    - Children of root: "{run_id}.0.0", "{run_id}.0.1", "{run_id}.0.2"
    - Grandchildren: "{run_id}.0.0.0", "{run_id}.0.0.1", etc.
    """
    
    def __init__(
        self,
        config: RunConfig,
        output_path: Optional[Union[str, Path]] = None
    ):
        self.config = config
        self.run_id = config.run_identity.run_id
        self.flush_every = config.output.flush_every
        
        if output_path:
            self.output_path = Path(output_path)
        else:
            output_dir = Path(config.output.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.output_path = output_dir / f"{self.run_id}.jsonl"
        
        self._file = None
        self._nodes_written = 0
        self._step_counter = 0
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def open(self) -> None:
        """Open the output file for writing."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, 'a')
    
    def close(self) -> None:
        """Close the output file, flushing any remaining data."""
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
    
    def write_node(self, node: NodeOutput) -> None:
        """
        Write a single node to the JSONL file.
        
        Flushes based on flush_every setting for crash recovery.
        """
        if self._file is None:
            raise RuntimeError("Writer not open. Use 'with' statement or call open().")
        
        line = json.dumps(node.to_jsonl_dict())
        self._file.write(line + '\n')
        self._nodes_written += 1
        
        if self._nodes_written % self.flush_every == 0:
            self._file.flush()
    
    def generate_node_id(self, parent_id: Optional[str], sibling_index: int) -> str:
        """
        Generate a hierarchical node ID.
        
        Args:
            parent_id: ID of parent node (None for root)
            sibling_index: Index among siblings (0, 1, 2, ...)
            
        Returns:
            Hierarchical node ID like "run_id.0.2.1"
        """
        if parent_id is None:
            return f"{self.run_id}.{sibling_index}"
        return f"{parent_id}.{sibling_index}"
    
    def next_step_index(self) -> int:
        """Get the next step index (generation order) and increment counter."""
        idx = self._step_counter
        self._step_counter += 1
        return idx
    
    @property
    def nodes_written(self) -> int:
        """Number of nodes written so far."""
        return self._nodes_written


def generate_run_id() -> str:
    """Generate a unique run ID from timestamp + random suffix."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"{ts}_{suffix}"


def read_jsonl(path: Union[str, Path]) -> list[dict]:
    """
    Read a JSONL file and return list of dicts.
    
    Handles partial files gracefully (skips malformed lines).
    """
    nodes = []
    path = Path(path)
    
    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                nodes.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}")
    
    return nodes


def iter_jsonl(path: Union[str, Path]):
    """
    Iterate over JSONL file, yielding one dict per line.
    
    Memory-efficient for large files.
    """
    path = Path(path)
    
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
