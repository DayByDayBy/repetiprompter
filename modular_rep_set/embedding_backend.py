#!/usr/bin/env python3
"""
Pluggable embedding backend for semantic drift analysis.

Supports multiple embedding providers:
- Ollama (default)
- sentence-transformers (optional)
- HuggingFace transformers (optional)

Uses numpy for vector operations.
"""

import abc
import math
from typing import List, Optional, Protocol, Union

import numpy as np


class EmbeddingBackend(abc.ABC):
    """Abstract base class for embedding backends."""
    
    @abc.abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding vector."""
        pass
    
    @abc.abstractmethod
    def batch_encode(self, texts: List[str]) -> List[np.ndarray]:
        """Encode multiple texts to embedding vectors."""
        pass


class OllamaEmbeddingBackend(EmbeddingBackend):
    """Ollama embedding backend."""
    
    def __init__(self, model: str = "nomic-embed-text"):
        import ollama
        self.model = model
        self.client = ollama
    
    def encode(self, text: str) -> np.ndarray:
        """Encode single text using Ollama."""
        response = self.client.embeddings(model=self.model, prompt=text)
        return np.array(response.embedding, dtype=np.float32)
    
    def batch_encode(self, texts: List[str]) -> List[np.ndarray]:
        """Encode multiple texts using Ollama (one by one)."""
        return [self.encode(text) for text in texts]


class SentenceTransformerBackend(EmbeddingBackend):
    """Sentence-transformers embedding backend."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: uv add sentence-transformers"
            ) from e
        
        self.model = SentenceTransformer(model_name)
    
    def encode(self, text: str) -> np.ndarray:
        """Encode single text using sentence-transformers."""
        return self.model.encode(text, convert_to_numpy=True)
    
    def batch_encode(self, texts: List[str]) -> List[np.ndarray]:
        """Encode multiple texts using sentence-transformers."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb for emb in embeddings]


class HuggingFaceTransformersBackend(EmbeddingBackend):
    """HuggingFace transformers embedding backend."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers and torch not installed. "
                "Install with: uv add transformers torch"
            ) from e
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def encode(self, text: str) -> np.ndarray:
        """Encode single text using HF transformers."""
        import torch
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy().flatten()
    
    def batch_encode(self, texts: List[str]) -> List[np.ndarray]:
        """Encode multiple texts using HF transformers."""
        import torch
        
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        embeddings_np = embeddings.cpu().numpy()
        return [emb for emb in embeddings_np]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors (1 - cosine similarity)."""
    return 1.0 - cosine_similarity(a, b)


def create_embedding_backend(
    backend_type: str = "ollama",
    model: Optional[str] = None,
) -> EmbeddingBackend:
    """Create embedding backend instance.
    
    Args:
        backend_type: Type of backend ("ollama", "sentence-transformers", "huggingface")
        model: Model name (uses default if None)
    
    Returns:
        EmbeddingBackend instance
    """
    if backend_type == "ollama":
        model = model or "nomic-embed-text"
        return OllamaEmbeddingBackend(model)
    elif backend_type == "sentence-transformers":
        model = model or "all-MiniLM-L6-v2"
        return SentenceTransformerBackend(model)
    elif backend_type == "huggingface":
        model = model or "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceTransformersBackend(model)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


if __name__ == "__main__":
    # Simple test
    backend = create_embedding_backend("ollama")
    
    test_texts = [
        "Hello world",
        "Goodbye world", 
        "The quick brown fox jumps over the lazy dog"
    ]
    
    print("Testing embedding backend...")
    for text in test_texts:
        embedding = backend.encode(text)
        print(f"Text: {text[:30]}... Shape: {embedding.shape}")
    
    # Test similarity
    emb1 = backend.encode("Hello world")
    emb2 = backend.encode("Hello there")
    emb3 = backend.encode("Completely different topic")
    
    sim_12 = cosine_similarity(emb1, emb2)
    sim_13 = cosine_similarity(emb1, emb3)
    
    print(f"Similarity('Hello world', 'Hello there'): {sim_12:.3f}")
    print(f"Similarity('Hello world', 'Completely different'): {sim_13:.3f}")
