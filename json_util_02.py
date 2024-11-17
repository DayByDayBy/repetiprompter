from typing import Dict, Any, List, Tuple, Set, Optional
from collections import defaultdict
import json
from pathlib import Path
import logging
from datetime import datetime
from dataclasses import dataclass

@dataclass
class SchemaStats:
    """statistics about a JSON schema"""
    total_keys: int
    max_depth: int
    array_counts: Dict[str, int]
    type_mismatches: Dict[str, Set[str]]
    null_fields: Set[str]
    optional_fields: Set[str]

class JSONSchemaAnalyzer:
    """analyzes and validates JSON data against expected schemas"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _get_type_name(self, value: Any) -> str:
        """Get a simplified type name for a value."""
        if value is None:
            return 'null'
        elif isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, int):
            return 'integer'
        elif isinstance(value, float):
            return 'number'
        elif isinstance(value, str):
            return 'string'
        elif isinstance(value, list):
            return 'array'
        elif isinstance(value, dict):
            return 'object'
        else:
            return type(value).__name__

    def infer_schema(self, data: Dict[str, Any], path: str = '') -> Dict[str, Any]:
        """
        recursively infer a schema from JSON data.
        
        args:
            data: JSON data to analyze
            path: Current path in the JSON structure
            
        returns:
            Inferred schema dictionary
        """
        if isinstance(data, dict):
            schema = {
                'type': 'object',
                'properties': {}
            }
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                schema['properties'][key] = self.infer_schema(value, current_path)
            return schema
            
        elif isinstance(data, list):
            if not data:
                return {'type': 'array', 'items': {}}
            
            # Infer schema from all array items
            item_schemas = [self.infer_schema(item, f"{path}[]") for item in data]
            # Merge schemas to find common structure
            return {
                'type': 'array',
                'items': self._merge_schemas(item_schemas)
            }
            
        else:
            return {'type': self._get_type_name(data)}

    def _merge_schemas(self, schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """merge multiple schemas into a single schema that accommodates all variations."""
        if not schemas:
            return {}
            
        # If all schemas are of the same type, merge them
        types = {s['type'] for s in schemas}
        if len(types) == 1:
            base_type = next(iter(types))
            if base_type == 'object':
                # Merge object properties
                all_properties = {}
                for schema in schemas:
                    for key, prop_schema in schema.get('properties', {}).items():
                        if key not in all_properties:
                            all_properties[key] = prop_schema
                        else:
                            all_properties[key] = self._merge_schemas([all_properties[key], prop_schema])
                return {
                    'type': 'object',
                    'properties': all_properties
                }
            elif base_type == 'array':
                # Merge array item schemas
                item_schemas = [s.get('items', {}) for s in schemas]
                return {
                    'type': 'array',
                    'items': self._merge_schemas(item_schemas)
                }
        
        # If types differ, create a union type
        return {'type': list(types)}

    def analyze_structure(self, data: Any, path: str = '') -> SchemaStats:
        """
        analyze the structure of JSON data and collect statistics.
        
        args:
            data: JSON data to analyze
            path: Current path in the JSON structure
            
        returns:
            SchemaStats object containing analysis results
        """
        stats = SchemaStats(
            total_keys=0,
            max_depth=0,
            array_counts={},
            type_mismatches={},
            null_fields=set(),
            optional_fields=set()
        )
        
        def _analyze_recursive(d: Any, current_path: str, depth: int) -> None:
            stats.max_depth = max(stats.max_depth, depth)
            
            if isinstance(d, dict):
                stats.total_keys += len(d)
                for key, value in d.items():
                    new_path = f"{current_path}.{key}" if current_path else key
                    
                    if value is None:
                        stats.null_fields.add(new_path)
                    
                    _analyze_recursive(value, new_path, depth + 1)
                    
            elif isinstance(d, list):
                stats.array_counts[current_path] = len(d)
                for item in d:
                    _analyze_recursive(item, f"{current_path}[]", depth + 1)
                    
        _analyze_recursive(data, path, 0)
        return stats

    def validate_against_schema(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str = ''
    ) -> List[str]:
        """
        validate JSON data against a schema and return list of violations.
        
        args:
            data: JSON data to validate
            schema: Schema to validate against
            path: Current path in the JSON structure
            
        returns:
            List of validation error messages
        """
        violations = []
        
        def _validate_recursive(d: Any, s: Dict[str, Any], current_path: str) -> None:
            expected_type = s.get('type')
            actual_type = self._get_type_name(d)
            
            # Handle union types
            if isinstance(expected_type, list):
                if actual_type not in expected_type:
                    violations.append(
                        f"type mismatch at {current_path}. "
                        f"expected one of {expected_type}, got {actual_type}"
                    )
                return
                
            if expected_type != actual_type:
                violations.append(
                    f"type mismatch at {current_path}. "
                    f"expected {expected_type}, got {actual_type}"
                )
                return
                
            if expected_type == 'object':
                required = s.get('required', list(s.get('properties', {}).keys()))
                for key in required:
                    if key not in d:
                        violations.append(f"missing required field {current_path}.{key}")
                        
                for key, value in d.items():
                    if key in s.get('properties', {}):
                        new_path = f"{current_path}.{key}" if current_path else key
                        _validate_recursive(value, s['properties'][key], new_path)
                        
            elif expected_type == 'array':
                if 'items' in s:
                    for i, item in enumerate(d):
                        _validate_recursive(item, s['items'], f"{current_path}[{i}]")
                        
        _validate_recursive(data, schema, path)
        return violations

    def generate_example(self, schema: Dict[str, Any]) -> Any:
        """
        generate example data that matches a schema.
        
        args:
            schema: schema to generate example from
            
        returns:
            Generated example data
        """
        type_name = schema.get('type')
        if isinstance(type_name, list):
            # For union types, use the first type
            type_name = type_name[0]
            
        if type_name == 'object':
            result = {}
            for key, prop_schema in schema.get('properties', {}).items():
                result[key] = self.generate_example(prop_schema)
            return result
            
        elif type_name == 'array':
            # Generate a small example array
            return [self.generate_example(schema['items']) for _ in range(2)]
            
        elif type_name == 'string':
            return "example"
        elif type_name == 'number':
            return 42.0
        elif type_name == 'integer':
            return 42
        elif type_name == 'boolean':
            return True
        elif type_name == 'null':
            return None
        else:
            return None

def example_usage():
    # Example JSON data with varying structures
    data1 = {
        "id": 1,
        "name": "example",
        "tags": ["a", "b", "c"],
        "metadata": {
            "created": "2024-01-01",
            "status": "active"
        }
    }
    
    data2 = {
        "id": 2,
        "name": None,  # different type
        "tags": [],    # empty array
        "metadata": {
            "created": "2024-01-02",
            "status": None,  # optional field
            "extra": "field" # additional field
        }
    }
    
    analyzer = JSONSchemaAnalyzer()
    
    # infer schema from first document
    schema = analyzer.infer_schema(data1)
    print("\nInferred Schema:")
    print(json.dumps(schema, indent=2))
    
    # analyze structure
    stats = analyzer.analyze_structure(data2)
    print("\nStructure Analysis:")
    print(f"Total Keys: {stats.total_keys}")
    print(f"Max Depth: {stats.max_depth}")
    print(f"Array Counts: {stats.array_counts}")
    print(f"Null Fields: {stats.null_fields}")
    
    # validate second document against schema
    violations = analyzer.validate_against_schema(data2, schema)
    print("\nValidation Results:")
    for violation in violations:
        print(f"- {violation}")
    
    # example data
    example = analyzer.generate_example(schema)
    print("\nGenerated Example:")
    print(json.dumps(example, indent=2))

if __name__ == '__main__':
    example_usage()