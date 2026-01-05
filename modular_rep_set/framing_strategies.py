"""Framing strategies for prompts - extracts legacy PREFIX/SUFFIX patterns into pluggable modules."""

from abc import ABC, abstractmethod
from typing import Optional

from .models import FramingStrategy as FramingStrategyEnum


class FramingStrategy(ABC):
    """Base class for framing strategies that wrap prompts with prefix/suffix."""
    
    name: str = "base"
    description: str = "Base framing strategy"
    
    @abstractmethod
    def apply_prefix(self, text: str) -> str:
        """Apply prefix to the prompt text."""
        pass
    
    @abstractmethod
    def apply_suffix(self, text: str) -> str:
        """Apply suffix to the prompt text."""
        pass
    
    def frame(self, text: str) -> tuple[str, str, str]:
        """
        Apply full framing to text.
        
        Returns:
            Tuple of (framed_text, prefix_used, suffix_used)
        """
        prefix = self.apply_prefix("")
        suffix = self.apply_suffix("")
        framed = f"{prefix}{text}{suffix}".strip()
        return framed, prefix.strip(), suffix.strip()


class SimpleStrategy(FramingStrategy):
    """No framing - pass through unchanged. Original repetiprompter_og.py style."""
    
    name = "simple"
    description = "No prefix or suffix - pure recursive prompting"
    
    def apply_prefix(self, text: str) -> str:
        return ""
    
    def apply_suffix(self, text: str) -> str:
        return ""


class LiarParadoxStrategy(FramingStrategy):
    """
    Liar's paradox injection from repetiprompter_delta.py.
    
    Creates logical tension by wrapping with contradictory truth claims.
    """
    
    name = "liar_paradox"
    description = "Wraps with 'the next sentence is false' / 'the previous sentence is true'"
    
    def apply_prefix(self, text: str) -> str:
        return "The next sentence is false. "
    
    def apply_suffix(self, text: str) -> str:
        return " The previous sentence is true."


class DiscussionStrategy(FramingStrategy):
    """
    Discussion prompt from repetiprompter_epsilon.py.
    
    Frames as ongoing conversation to encourage engagement.
    """
    
    name = "discussion"
    description = "Frames as ongoing discussion - 'welcome to the ongoing discussion... what do you think?'"
    
    def apply_prefix(self, text: str) -> str:
        return "Welcome to the ongoing discussion. The previous participant said: \""
    
    def apply_suffix(self, text: str) -> str:
        return "\" What do you think?"


class RephraseStrategy(FramingStrategy):
    """
    Rephrase as question from repetiprompter_zeta.py.
    
    Encourages transformation of statements into questions.
    """
    
    name = "rephrase"
    description = "Asks to rephrase content as a question"
    
    def apply_prefix(self, text: str) -> str:
        return "Rephrase the following as a question: \""
    
    def apply_suffix(self, text: str) -> str:
        return "\""


class EchoStrategy(FramingStrategy):
    """
    Echo/attribution from repetiprompter_gamma.py.
    
    Frames as reported speech.
    """
    
    name = "echo"
    description = "Frames as reported speech - 'Say I said...'"
    
    def apply_prefix(self, text: str) -> str:
        return "Say I said: \""
    
    def apply_suffix(self, text: str) -> str:
        return "\""


class CustomStrategy(FramingStrategy):
    """Custom prefix/suffix defined by user."""
    
    name = "custom"
    description = "User-defined prefix and suffix"
    
    def __init__(self, prefix: str = "", suffix: str = ""):
        self._prefix = prefix
        self._suffix = suffix
    
    def apply_prefix(self, text: str) -> str:
        return self._prefix
    
    def apply_suffix(self, text: str) -> str:
        return self._suffix


# Registry of available strategies
_STRATEGY_REGISTRY: dict[str, type[FramingStrategy]] = {
    "simple": SimpleStrategy,
    "liar_paradox": LiarParadoxStrategy,
    "discussion": DiscussionStrategy,
    "rephrase": RephraseStrategy,
    "echo": EchoStrategy,
    "custom": CustomStrategy,
}


def get_strategy(
    name: str | FramingStrategyEnum,
    custom_prefix: str = "",
    custom_suffix: str = ""
) -> FramingStrategy:
    """
    Get a framing strategy by name.
    
    Args:
        name: Strategy name or enum
        custom_prefix: Prefix for custom strategy
        custom_suffix: Suffix for custom strategy
        
    Returns:
        Instantiated strategy
        
    Raises:
        ValueError: If strategy name not found
    """
    if isinstance(name, FramingStrategyEnum):
        name = name.value
    
    if name not in _STRATEGY_REGISTRY:
        available = ", ".join(_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown framing strategy: {name}. Available: {available}")
    
    strategy_cls = _STRATEGY_REGISTRY[name]
    
    if name == "custom":
        return strategy_cls(prefix=custom_prefix, suffix=custom_suffix)
    
    return strategy_cls()


def list_strategies() -> list[dict]:
    """List all available strategies with descriptions."""
    return [
        {"name": name, "description": cls.description}
        for name, cls in _STRATEGY_REGISTRY.items()
    ]
