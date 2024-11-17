import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
import datetime
import logging

TIME_STAMP = datetime.now().strftime("%Y%m%d_%H%M")


class JSONUtils:
    """utility class for handling JSON operations with enhanced features."""
    
    def __init__(self, indent: int = 2, encoding: str = 'utf-8'):
        self.indent = indent
        self.encoding = encoding
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up basic logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _ensure_directory(self, filepath: Union[str, Path]) -> Path:
        """ensure the directory exists for the given filepath."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        return filepath
    
    def _serialize_default(self, obj: Any) -> str:
        """custom serialization for non-JSON serializable objects."""
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        return str(obj)
    
    def save_json(
        self,
        data: Dict,
        filepath: Union[str, Path],
        backup: bool = True,
        pretty: bool = True
    ) -> bool:
        """
        save nested dictionary to JSON file with error handling and backup option

        args:
            data: Dictionary to save
            filepath: Path to save the JSON file
            backup: If True, create a backup of existing file
            pretty: If True, format with indentation and newlines

        returns:
            bool: True if save was successful, False otherwise
        """
        try:
            filepath = self._ensure_directory(filepath)
            
            # Create backup if requested and file exists
            if backup and filepath.exists():
                backup_path = filepath.with_suffix(f'.backup_{datetime.datetime.now():%Y%m%d_%H%M%S}.json')
                filepath.rename(backup_path)
                self.logger.info(f"created backup at {backup_path}")
            
            with open(filepath, 'w', encoding=self.encoding) as f:
                if pretty:
                    json.dump(
                        data,
                        f,
                        indent=self.indent,
                        default=self._serialize_default,
                        ensure_ascii=False
                    )
                else:
                    json.dump(
                        data,
                        f,
                        default=self._serialize_default,
                        ensure_ascii=False
                    )
                    
            self.logger.info(f"successfully saved JSON to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"error saving JSON to {filepath}: {str(e)}")
            return False
    
    def save_json_tree(
        self,
        data: Dict,
        filepath: Union[str, Path],
        backup: bool = True
    ) -> bool:
        """
        save nested dictionary to JSON file, preserving tree structure with custom formatting.

        args:
            data: Dict to save
            filepath: Path to save the JSON file
            backup: If True, create a backup of existing file

        returns:
            bool: True if save was successful, False otherwise
        """
        def _format_tree(obj: Any, indent: int = 0) -> str:
            if isinstance(obj, dict):
                lines = ['{\n']
                items = list(obj.items())
                for i, (key, value) in enumerate(items):
                    lines.append('  ' * (indent + 1) + f'"{key}": ')
                    lines.append(_format_tree(value, indent + 1))
                    if i < len(items) - 1:
                        lines[-1] = lines[-1] + ','
                    lines[-1] = lines[-1] + '\n'
                lines.append('  ' * indent + '}')
                return ''.join(lines)
                
            elif isinstance(obj, list):
                if not obj:
                    return '[]'
                lines = ['[\n']
                for i, item in enumerate(obj):
                    lines.append('  ' * (indent + 1))
                    lines.append(_format_tree(item, indent + 1))
                    if i < len(obj) - 1:
                        lines[-1] = lines[-1] + ','
                    lines[-1] = lines[-1] + '\n'
                lines.append('  ' * indent + ']')
                return ''.join(lines)
                
            else:
                if isinstance(obj, str):
                    return f'"{obj}"'
                return json.dumps(obj, default=self._serialize_default)

        try:
            filepath = self._ensure_directory(filepath)
            
            if backup and filepath.exists():
                backup_path = filepath.with_suffix(f'.backup_{datetime.datetime.now():%Y%m%d_%H%M%S}.json')
                filepath.rename(backup_path)
                self.logger.info(f"Created backup at {backup_path}")
            
            with open(filepath, 'w', encoding=self.encoding) as f:
                f.write(_format_tree(data))
                
            self.logger.info(f"Successfully saved JSON tree to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving JSON tree to {filepath}: {str(e)}")
            return False

    @staticmethod
    def load_json(filepath: Union[str, Path], default: Optional[Dict] = None) -> Optional[Dict]:
        """
        Load JSON file with error handling.

        Args:
            filepath: Path to JSON file
            default: Default value to return if loading fails

        Returns:
            Loaded JSON data or default value if loading fails
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading JSON from {filepath}: {str(e)}")
            return default

if __name__ == '__main__':
    # Example usage
    json_utils = JSONUtils()
    
    test_data = {
        'string': 'hello',
        'number': 42,
        'nested': {
            'list': [1, 2, 3],
            'dict': {'a': 1, 'b': 2},
            'datetime': datetime.datetime.now(),
            'set': {1, 2, 3}
        }
    }
    
    # saves twice
    json_utils.save_json(test_data, f'output/data_{TIME_STAMP}.json')
    json_utils.save_json_tree(test_data, f'output/data_tree_{TIME_STAMP}.json')
    
    # Load and verify
    loaded_data = json_utils.load_json('output/data.json')
    if loaded_data:
        print("Successfully loaded JSON data")