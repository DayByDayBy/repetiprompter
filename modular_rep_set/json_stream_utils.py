from pathlib import Path
from typing import Dict, Any, Union, Optional
import json
from datetime import datetime
import logging

class MLOutputWriter:
    """handles JSON output writing for machine learning experiments"""
    
    def __init__(
        self,
        base_path: Union[str, Path] = "outputs",
        model_name: Optional[str] = None,
        create_timestamp_dir: bool = True
    ):
        self.base_path = Path(base_path)
        self.model_name = model_name
        self.create_timestamp_dir = create_timestamp_dir
        self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _get_output_path(self, filename: Optional[str] = None) -> Path:
        """constructs output path with optional timestamp directory"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        if self.create_timestamp_dir:
            # create timestamp directory
            output_dir = self.base_path / timestamp
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = self.base_path
            
        if filename:
            return output_dir / filename
        else:
            # generate filename with model name if provided
            base_name = f"{self.model_name}_" if self.model_name else ""
            return output_dir / f"{base_name}{timestamp}.json"
    
    def init_experiment(
        self,
        metadata: Dict[str, Any],
        filepath: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Initialize experiment output file with metadata.
        
        Args:
            metadata: Dictionary containing experiment metadata
            filepath: Optional specific filepath to use
            
        Returns:
            Path object pointing to created file
        """
        output_path = Path(filepath) if filepath else self._get_output_path()
        
        try:
            with open(output_path, 'w') as f:
                json.dump({
                    'metadata': metadata,
                    'timestamps': {
                        'start': datetime.now().isoformat(),
                    },
                    'iterations': []
                }, f, indent=2)
                f.write('\n')
                f.flush()
            
            self.logger.info(f"Initialized experiment output at {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error initializing experiment file: {e}")
            raise
    
    def append_iteration(
        self,
        filepath: Union[str, Path],
        iteration_data: Dict[str, Any],
        iteration_num: Optional[int] = None
    ):
        """
        Append iteration data to experiment file.
        
        Args:
            filepath: Path to experiment file
            iteration_data: Dictionary containing iteration results
            iteration_num: Optional iteration number to include
        """
        try:
            if iteration_num is not None:
                iteration_data = {
                    'iteration': iteration_num,
                    'timestamp': datetime.now().isoformat(),
                    **iteration_data
                }
            
            with open(filepath, 'a') as f:
                json.dump(iteration_data, f)
                f.write('\n')
                f.flush()
                
        except Exception as e:
            self.logger.error(f"Error appending iteration data: {e}")
            raise
    
    def finalize_experiment(
        self,
        filepath: Union[str, Path],
        final_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add final metadata and clean up experiment file.
        
        Args:
            filepath: Path to experiment file
            final_metadata: Optional additional metadata to include
        """
        try:
            # Read existing content
            with open(filepath, 'r') as f:
                data = [json.loads(line) for line in f if line.strip()]
            
            # Update with final metadata
            if final_metadata:
                data[0]['final_metadata'] = final_metadata
            data[0]['timestamps']['end'] = datetime.now().isoformat()
            
            # Rewrite file with proper structure
            with open(filepath, 'w') as f:
                json.dump({
                    'metadata': data[0]['metadata'],
                    'timestamps': data[0]['timestamps'],
                    'iterations': data[1:],
                    **(({'final_metadata': final_metadata} if final_metadata else {}))
                }, f, indent=2)
                
            self.logger.info(f"Finalized experiment output at {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error finalizing experiment file: {e}")
            raise
