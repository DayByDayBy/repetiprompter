"""Reminder regime - random re-injection of original prompt with logging."""

import random
from dataclasses import dataclass
from typing import Optional

from .models import ReminderConfig


@dataclass
class ReminderResult:
    """Result of a reminder check."""
    fired: bool
    content: Optional[str] = None


class ReminderRegime:
    """
    Handles random re-injection of original prompt.
    
    This was abandoned in epsilon because firings weren't logged.
    Now every firing is tracked and returned for logging in output.
    """
    
    def __init__(
        self,
        config: ReminderConfig,
        initial_prompt: str
    ):
        self.enabled = config.enabled
        self.probability = config.probability
        self._initial_prompt = initial_prompt
        
        if config.content == "original":
            self._reminder_content = initial_prompt
        else:
            self._reminder_content = config.content
    
    def check(self) -> ReminderResult:
        """
        Check if reminder should fire this step.
        
        Returns:
            ReminderResult with fired=True/False and content if fired
        """
        if not self.enabled:
            return ReminderResult(fired=False)
        
        if random.random() < self.probability:
            return ReminderResult(fired=True, content=self._reminder_content)
        
        return ReminderResult(fired=False)
    
    def apply_to_prompt(self, prompt: str, result: ReminderResult) -> str:
        """
        Apply reminder to prompt if it fired.
        
        Args:
            prompt: The current prompt text
            result: ReminderResult from check()
            
        Returns:
            Modified prompt with reminder prepended (if fired) or original
        """
        if result.fired and result.content:
            return f'("{result.content}")\n\n{prompt}'
        return prompt
    
    @property
    def reminder_content(self) -> str:
        """The content that would be injected if reminder fires."""
        return self._reminder_content


def create_reminder_regime(
    enabled: bool = False,
    probability: float = 0.3,
    content: str = "original",
    initial_prompt: str = ""
) -> ReminderRegime:
    """Convenience factory for creating ReminderRegime."""
    config = ReminderConfig(
        enabled=enabled,
        probability=probability,
        content=content
    )
    return ReminderRegime(config, initial_prompt)
