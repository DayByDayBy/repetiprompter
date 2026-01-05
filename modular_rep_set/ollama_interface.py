"""Ollama interface with correct token counting from Ollama's native metrics."""

import time
from dataclasses import dataclass
from typing import Optional
import ollama


@dataclass
class GenerationResult:
    """Result from a single generation, including Ollama's metrics."""
    response: str
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    prompt_eval_duration_ms: Optional[int] = None
    eval_duration_ms: Optional[int] = None
    total_duration_ms: Optional[int] = None
    model: Optional[str] = None


class OllamaClient:
    """
    Clean interface to Ollama that uses native token counts.
    
    Fixes the tiktoken issue by using Ollama's reported counts
    (prompt_eval_count, eval_count) which are accurate for the
    actual model being used.
    """
    
    def __init__(
        self,
        model_name: str = "stablelm2:zephyr",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        seed: Optional[int] = None
    ) -> GenerationResult:
        """
        Generate a response from Ollama.
        
        Args:
            prompt: The prompt text
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter
            repeat_penalty: Repetition penalty
            seed: Optional seed for reproducibility
            
        Returns:
            GenerationResult with response and metrics
            
        Raises:
            RuntimeError: If all retries fail
        """
        options = {
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
        }
        if seed is not None:
            options["seed"] = seed
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options=options
                )
                return self._parse_result(result)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        raise RuntimeError(f"Ollama generation failed after {self.max_retries} retries: {last_error}")
    
    def _parse_result(self, result: dict) -> GenerationResult:
        """Parse Ollama response into GenerationResult with metrics."""
        return GenerationResult(
            response=result.get("response", ""),
            prompt_eval_count=result.get("prompt_eval_count"),
            eval_count=result.get("eval_count"),
            prompt_eval_duration_ms=self._ns_to_ms(result.get("prompt_eval_duration")),
            eval_duration_ms=self._ns_to_ms(result.get("eval_duration")),
            total_duration_ms=self._ns_to_ms(result.get("total_duration")),
            model=result.get("model", self.model_name)
        )
    
    @staticmethod
    def _ns_to_ms(ns: Optional[int]) -> Optional[int]:
        """Convert nanoseconds to milliseconds."""
        if ns is None:
            return None
        return ns // 1_000_000
    
    def check_model_available(self) -> bool:
        """Check if the configured model is available."""
        try:
            models = ollama.list()
            model_names = [m.get("name", "") for m in models.get("models", [])]
            return any(self.model_name in name for name in model_names)
        except Exception:
            return False
    
    def list_models(self) -> list[str]:
        """List available models."""
        try:
            models = ollama.list()
            return [m.get("name", "") for m in models.get("models", [])]
        except Exception:
            return []
