"""Temperature regime - controls how temperature changes over depth/steps."""

from abc import ABC, abstractmethod
from typing import Optional

from .models import TemperatureRegimeConfig, TemperatureRegimeType


class TemperatureRegime(ABC):
    """Base class for temperature regimes."""
    
    @abstractmethod
    def get_temperature(self, depth: int, step: int = 0) -> float:
        """
        Get temperature for a given depth/step.
        
        Args:
            depth: Current depth in tree (0 = root)
            step: Current step index (generation order)
            
        Returns:
            Temperature value to use
        """
        pass


class StaticTemperature(TemperatureRegime):
    """Same temperature for all nodes."""
    
    def __init__(self, temperature: float = 0.8):
        self.temperature = temperature
    
    def get_temperature(self, depth: int, step: int = 0) -> float:
        return self.temperature


class RampTemperature(TemperatureRegime):
    """
    Linear ramp from min to max over depth.
    
    Legacy zeta behavior - increases chaos as depth increases.
    """
    
    def __init__(
        self,
        min_temperature: float = 0.5,
        max_temperature: float = 1.2,
        max_depth: int = 5
    ):
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.max_depth = max_depth
    
    def get_temperature(self, depth: int, step: int = 0) -> float:
        if self.max_depth <= 0:
            return self.max_temperature
        
        progress = min(depth / self.max_depth, 1.0)
        return self.min_temperature + progress * (self.max_temperature - self.min_temperature)


class ScheduleTemperature(TemperatureRegime):
    """Explicit temperature per depth level."""
    
    def __init__(
        self,
        temperatures: dict[int, float],
        default: float = 1.0
    ):
        self.temperatures = temperatures
        self.default = default
    
    def get_temperature(self, depth: int, step: int = 0) -> float:
        return self.temperatures.get(depth, self.default)


def create_temperature_regime(config: TemperatureRegimeConfig) -> TemperatureRegime:
    """
    Factory to create appropriate temperature regime from config.
    
    Args:
        config: TemperatureRegimeConfig from run config
        
    Returns:
        Appropriate TemperatureRegime instance
    """
    if config.type == TemperatureRegimeType.STATIC:
        return StaticTemperature(temperature=config.static.temperature)
    
    elif config.type == TemperatureRegimeType.RAMP:
        return RampTemperature(
            min_temperature=config.ramp.min_temperature,
            max_temperature=config.ramp.max_temperature,
            max_depth=config.ramp.max_depth
        )
    
    elif config.type == TemperatureRegimeType.SCHEDULE:
        return ScheduleTemperature(
            temperatures=config.schedule.temperatures,
            default=config.schedule.default
        )
    
    raise ValueError(f"Unknown temperature regime type: {config.type}")
