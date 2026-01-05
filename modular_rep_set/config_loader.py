"""Configuration loader for repetiprompter - loads and validates YAML configs."""

from pathlib import Path
from typing import Union, Any, Dict, Optional
import yaml

from .models import RunConfig


def load_config(config_path: Union[str, Path]) -> RunConfig:
    """
    Load and validate a YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated RunConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        raise ValueError(f"Empty config file: {path}")
    
    return RunConfig.model_validate(raw_config)


def apply_overrides(config: RunConfig, overrides: Dict[str, Any]) -> RunConfig:
    """
    Apply dotted-notation overrides to a config.
    
    Args:
        config: Existing RunConfig instance
        overrides: Dict of dotted keys to values (e.g., {"model.temperature": 0.9})
        
    Returns:
        New RunConfig with overrides applied
    """
    config_dict = config.model_dump()
    
    for key, value in overrides.items():
        _set_nested(config_dict, key.split('.'), value)
    
    return RunConfig.model_validate(config_dict)


def _set_nested(d: dict, keys: list, value: Any) -> None:
    """Set a value in a nested dict using a list of keys."""
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def parse_override(override_str: str) -> tuple[str, Any]:
    """
    Parse an override string like "model.temperature=0.9".
    
    Args:
        override_str: String in format "key=value"
        
    Returns:
        Tuple of (key, parsed_value)
    """
    if '=' not in override_str:
        raise ValueError(f"Invalid override format: {override_str}. Expected 'key=value'")
    
    key, value_str = override_str.split('=', 1)
    key = key.strip()
    value_str = value_str.strip()
    
    value = _parse_value(value_str)
    return key, value


def _parse_value(value_str: str) -> Any:
    """Parse a string value to appropriate Python type."""
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
    if value_str.lower() == 'null' or value_str.lower() == 'none':
        return None
    
    try:
        return int(value_str)
    except ValueError:
        pass
    
    try:
        return float(value_str)
    except ValueError:
        pass
    
    if value_str.startswith('"') and value_str.endswith('"'):
        return value_str[1:-1]
    if value_str.startswith("'") and value_str.endswith("'"):
        return value_str[1:-1]
    
    return value_str


def create_minimal_config(
    initial_prompt: str,
    model_name: str = "stablelm2:zephyr",
    mode: str = "chain",
    steps: int = 10,
    depth: int = 5,
    branching_factor: int = 3,
    output_dir: str = "./runs"
) -> RunConfig:
    """
    Create a minimal config programmatically.
    
    Useful for quick runs without a YAML file.
    """
    config_dict = {
        "model": {"model_name": model_name},
        "prompting": {"initial_prompt": initial_prompt},
        "topology": {
            "mode": mode,
            "chain": {"steps": steps},
            "tree": {"depth": depth, "branching_factor": branching_factor}
        },
        "output": {"output_dir": output_dir}
    }
    return RunConfig.model_validate(config_dict)


def save_config(config: RunConfig, path: Union[str, Path]) -> None:
    """Save a config to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.model_dump()
    if config_dict.get('run_identity', {}).get('created_at'):
        config_dict['run_identity']['created_at'] = config_dict['run_identity']['created_at'].isoformat()
    
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
