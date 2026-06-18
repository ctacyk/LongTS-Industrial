"""
Extended events for the time_blender library.
This module provides additional generic events that can be combined to create complex time series patterns.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from scipy import signal

from time_blender.core import Event, LambdaEvent, ConstantEvent, Invariant
from time_blender.random_events import NormalEvent, BernoulliEvent, PoissonEvent
from time_blender.deterministic_events import WaveEvent, WalkEvent, ClipEvent, IdentityEvent
from time_blender.coordination_events import PastEvent, SeasonalEvent, Choice, CumulativeEvent


class RegimeSwitchingEvent(Event):
    """
    An event that switches between different regimes or states based on probabilities or conditions.
    """
    
    def __init__(self, regimes: List[Event], transition_probabilities: List[List[float]] = None,
                 initial_regime: int = 0, switching_condition: Event = None,
                 name: str = None, parallel_events: List[Event] = None):
        """
        Initialize a regime switching event.
        
        :param regimes: List of events representing different regimes
        :param transition_probabilities: Transition probability matrix between regimes
        :param initial_regime: Initial regime index
        :param switching_condition: Event that determines when to switch regimes
        :param name: Name of the event
        :param parallel_events: Parallel events
        """
        super().__init__(name, parallel_events)
        
        self.regimes = regimes
        self.transition_probabilities = transition_probabilities
        self.current_regime = initial_regime
        self.switching_condition = switching_condition
        self._last_switch_time = None
        
        # Initialize transition probabilities if not provided
        if self.transition_probabilities is None and len(regimes) > 0:
            n = len(regimes)
            # Uniform transition probabilities
            self.transition_probabilities = [[1.0/n for _ in range(n)] for _ in range(n)]
        
        # Add regimes as causal parameters
        for regime in self.regimes:
            if isinstance(regime, Event):
                self._causal_parameters.append(regime)

    def _execute(self, t, i):
        # Check if we should switch regimes
        should_switch = False
        
        if self.switching_condition is not None:
            # Switch based on condition
            should_switch = bool(self.switching_condition.execute(t))
        elif self.transition_probabilities is not None and len(self.transition_probabilities) > 0:
            # Switch based on probabilities
            if np.random.random() < 0.01:  # 1% chance to check for switch at each step
                # Determine next regime based on transition probabilities
                current_probs = self.transition_probabilities[self.current_regime]
                next_regime = np.random.choice(len(self.regimes), p=current_probs)
                if next_regime != self.current_regime:
                    self.current_regime = next_regime
                    should_switch = True
        
        # Execute current regime
        return self.regimes[self.current_regime].execute(t)


class ExternalFactorEvent(Event):
    """
    An event that models the impact of external factors on a time series.
    """
    
    def __init__(self, base_event: Event, external_factors: List[Event], 
                 coefficients: List[float] = None, interaction_type: str = "additive",
                 name: str = None, parallel_events: List[Event] = None):
        """
        Initialize an external factor event.
        
        :param base_event: The base event to which external factors are applied
        :param external_factors: List of external factor events
        :param coefficients: Coefficients for each external factor
        :param interaction_type: How factors interact with base - "additive" or "multiplicative"
        :param name: Name of the event
        :param parallel_events: Parallel events
        """
        super().__init__(name, parallel_events)
        
        self.base_event = base_event
        self.external_factors = external_factors
        self.interaction_type = interaction_type
        
        # Initialize coefficients if not provided
        if coefficients is None:
            self.coefficients = [1.0] * len(external_factors)
        else:
            self.coefficients = coefficients
            
        # Add as causal parameters
        self._causal_parameters.append(base_event)
        for factor in external_factors:
            if isinstance(factor, Event):
                self._causal_parameters.append(factor)

    def _execute(self, t, i):
        # Execute base event
        base_value = self.base_event.execute(t)
        
        # Apply external factors
        if self.interaction_type == "additive":
            result = base_value
            for factor, coeff in zip(self.external_factors, self.coefficients):
                factor_value = factor.execute(t)
                result += coeff * factor_value
        elif self.interaction_type == "multiplicative":
            result = base_value
            for factor, coeff in zip(self.external_factors, self.coefficients):
                factor_value = factor.execute(t)
                result *= (1.0 + coeff * factor_value)
        else:
            result = base_value
            
        return result


class ComplexWaveEvent(Event):
    """
    A complex wave event that can combine multiple waveforms.
    """
    
    def __init__(self, frequencies: List[float], amplitudes: List[float], 
                 phases: List[float] = None, wave_types: List[str] = None,
                 combination_method: str = "add",
                 name: str = None, parallel_events: List[Event] = None):
        """
        Initialize a complex wave event.
        
        :param frequencies: List of frequencies for each wave component
        :param amplitudes: List of amplitudes for each wave component
        :param phases: List of phases for each wave component
        :param wave_types: List of wave types - "sine", "cosine", "square", "sawtooth", "triangle"
        :param combination_method: How to combine waves - "add" or "multiply"
        :param name: Name of the event
        :param parallel_events: Parallel events
        """
        super().__init__(name, parallel_events)
        
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        self.combination_method = combination_method
        
        # Initialize phases if not provided
        if phases is None:
            self.phases = [0.0] * len(frequencies)
        else:
            self.phases = phases
            
        # Initialize wave types if not provided
        if wave_types is None:
            self.wave_types = ["sine"] * len(frequencies)
        else:
            self.wave_types = wave_types

    def _execute(self, t, i):
        # Convert time index to continuous time (assuming hourly by default)
        time_sec = i * 3600  # Convert to seconds
        
        # Calculate each wave component
        components = []
        for freq, amp, phase, wave_type in zip(self.frequencies, self.amplitudes, self.phases, self.wave_types):
            # Convert frequency from cycles per period to angular frequency
            angular_freq = 2 * np.pi * freq
            
            # Calculate phase in radians
            phase_rad = 2 * np.pi * phase
            
            # Generate wave based on type
            if wave_type == "sine":
                value = amp * np.sin(angular_freq * time_sec + phase_rad)
            elif wave_type == "cosine":
                value = amp * np.cos(angular_freq * time_sec + phase_rad)
            elif wave_type == "square":
                value = amp * signal.square(angular_freq * time_sec + phase_rad)
            elif wave_type == "sawtooth":
                value = amp * signal.sawtooth(angular_freq * time_sec + phase_rad)
            elif wave_type == "triangle":
                value = amp * signal.sawtooth(angular_freq * time_sec + phase_rad, 0.5)
            else:
                value = amp * np.sin(angular_freq * time_sec + phase_rad)
                
            components.append(value)
        
        # Combine components
        if self.combination_method == "add":
            result = sum(components)
        elif self.combination_method == "multiply":
            result = 1.0
            for comp in components:
                result *= comp
        else:
            result = sum(components)
            
        return result


class StochasticProcessEvent(Event):
    """
    A generic stochastic process event that can model various random processes.
    """
    
    def __init__(self, process_type: str = "ou", 
                 mean_reversion_speed: float = 0.1, 
                 long_term_mean: float = 0.0,
                 volatility: float = 1.0,
                 initial_value: float = 0.0,
                 jump_intensity: float = 0.0,
                 jump_mean: float = 0.0,
                 jump_std: float = 1.0,
                 name: str = None, parallel_events: List[Event] = None):
        """
        Initialize a stochastic process event.
        
        :param process_type: Type of process - "ou" (Ornstein-Uhlenbeck), "gbm" (Geometric Brownian Motion),
                            "jump" (Jump diffusion), "mean" (Mean reverting with jumps)
        :param mean_reversion_speed: Speed of mean reversion (for OU and mean processes)
        :param long_term_mean: Long-term mean (for OU and mean processes)
        :param volatility: Volatility parameter
        :param initial_value: Initial value of the process
        :param jump_intensity: Intensity of jumps (for jump processes)
        :param jump_mean: Mean of jump sizes (for jump processes)
        :param jump_std: Standard deviation of jump sizes (for jump processes)
        :param name: Name of the event
        :param parallel_events: Parallel events
        """
        super().__init__(name, parallel_events)
        
        self.process_type = process_type
        self.mean_reversion_speed = mean_reversion_speed
        self.long_term_mean = long_term_mean
        self.volatility = volatility
        self.initial_value = initial_value
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        
        # Current value of the process
        self.current_value = initial_value
        self._last_time = 0

    def _execute(self, t, i):
        # Time step (assuming hourly data)
        dt = 1.0 / 365.0  # Approximate annualized time step
        
        # Generate random components
        brownian_increment = np.random.normal(0, np.sqrt(dt))
        
        # Update process based on type
        if self.process_type == "ou":
            # Ornstein-Uhlenbeck process
            drift = self.mean_reversion_speed * (self.long_term_mean - self.current_value) * dt
            diffusion = self.volatility * brownian_increment
            self.current_value += drift + diffusion
            
        elif self.process_type == "gbm":
            # Geometric Brownian Motion
            drift = self.long_term_mean * self.current_value * dt
            diffusion = self.volatility * self.current_value * brownian_increment
            self.current_value += drift + diffusion
            
        elif self.process_type == "jump":
            # Jump diffusion process
            drift = self.long_term_mean * self.current_value * dt
            diffusion = self.volatility * self.current_value * brownian_increment
            
            # Add jump component
            jump = 0.0
            if np.random.random() < self.jump_intensity * dt:
                jump = np.random.normal(self.jump_mean, self.jump_std)
                
            self.current_value += drift + diffusion + jump
            
        elif self.process_type == "mean":
            # Mean reverting process with jumps
            drift = self.mean_reversion_speed * (self.long_term_mean - self.current_value) * dt
            diffusion = self.volatility * brownian_increment
            
            # Add jump component
            jump = 0.0
            if np.random.random() < self.jump_intensity * dt:
                jump = np.random.normal(self.jump_mean, self.jump_std)
                
            self.current_value += drift + diffusion + jump
            
        self._last_time = i
        return self.current_value


class RegimeDependentEvent(Event):
    """
    An event whose behavior changes based on different regimes or states.
    """
    
    def __init__(self, regime_indicator: Event, regime_parameters: Dict[Any, Dict[str, Any]],
                 base_event_factory: callable, name: str = None, parallel_events: List[Event] = None):
        """
        Initialize a regime dependent event.
        
        :param regime_indicator: Event that indicates the current regime
        :param regime_parameters: Dictionary mapping regime values to parameter dictionaries
        :param base_event_factory: Function that creates base events given parameters
        :param name: Name of the event
        :param parallel_events: Parallel events
        """
        super().__init__(name, parallel_events)
        
        self.regime_indicator = regime_indicator
        self.regime_parameters = regime_parameters
        self.base_event_factory = base_event_factory
        
        # Current regime and event
        self.current_regime = None
        self.current_event = None
        
        # Add regime indicator as causal parameter
        self._causal_parameters.append(regime_indicator)

    def _execute(self, t, i):
        # Get current regime
        regime = self.regime_indicator.execute(t)
        
        # Check if regime has changed
        if regime != self.current_regime:
            # Create new event for this regime
            if regime in self.regime_parameters:
                params = self.regime_parameters[regime]
                self.current_event = self.base_event_factory(**params)
                self._causal_parameters.append(self.current_event)
            else:
                # Use default parameters
                self.current_event = self.base_event_factory()
                self._causal_parameters.append(self.current_event)
                
            self.current_regime = regime
            
        # Execute current event
        if self.current_event is not None:
            return self.current_event.execute(t)
        else:
            return 0.0


class NonlinearTransformationEvent(Event):
    """
    An event that applies nonlinear transformations to another event.
    """
    
    def __init__(self, base_event: Event, transformation_type: str = "log",
                 transformation_params: Dict[str, Any] = None,
                 name: str = None, parallel_events: List[Event] = None):
        """
        Initialize a nonlinear transformation event.
        
        :param base_event: The event to transform
        :param transformation_type: Type of transformation - "log", "exp", "power", "sigmoid", "tanh", "relu"
        :param transformation_params: Additional parameters for the transformation
        :param name: Name of the event
        :param parallel_events: Parallel events
        """
        super().__init__(name, parallel_events)
        
        self.base_event = base_event
        self.transformation_type = transformation_type
        self.transformation_params = transformation_params or {}
        
        # Add base event as causal parameter
        self._causal_parameters.append(base_event)

    def _execute(self, t, i):
        # Get base value
        base_value = self.base_event.execute(t)
        
        # Apply transformation
        if self.transformation_type == "log":
            # Add small constant to avoid log(0)
            epsilon = self.transformation_params.get("epsilon", 1e-8)
            result = np.log(np.abs(base_value) + epsilon)
        elif self.transformation_type == "exp":
            # Limit exponent to prevent overflow
            max_exp = self.transformation_params.get("max_exp", 10.0)
            result = np.exp(np.clip(base_value, -max_exp, max_exp))
        elif self.transformation_type == "power":
            power = self.transformation_params.get("power", 2.0)
            result = np.power(base_value, power)
        elif self.transformation_type == "sigmoid":
            result = 1.0 / (1.0 + np.exp(-base_value))
        elif self.transformation_type == "tanh":
            result = np.tanh(base_value)
        elif self.transformation_type == "relu":
            result = np.maximum(0.0, base_value)
        elif self.transformation_type == "threshold":
            threshold = self.transformation_params.get("threshold", 0.0)
            result = np.where(base_value > threshold, base_value, 0.0)
        else:
            result = base_value
            
        return result


class ConditionalEvent(Event):
    """
    An event that applies different behaviors based on conditions.
    """
    
    def __init__(self, condition_event: Event, true_event: Event, false_event: Event = None,
                 name: str = None, parallel_events: List[Event] = None):
        """
        Initialize a conditional event.
        
        :param condition_event: Event that determines the condition
        :param true_event: Event to execute when condition is true
        :param false_event: Event to execute when condition is false (optional, defaults to 0)
        :param name: Name of the event
        :param parallel_events: Parallel events
        """
        super().__init__(name, parallel_events)
        
        self.condition_event = condition_event
        self.true_event = true_event
        self.false_event = false_event or ConstantEvent(0.0)
        
        # Add as causal parameters
        self._causal_parameters.append(condition_event)
        self._causal_parameters.append(true_event)
        self._causal_parameters.append(false_event)

    def _execute(self, t, i):
        # Evaluate condition
        condition = self.condition_event.execute(t)
        
        # Execute appropriate event
        if bool(condition):
            return self.true_event.execute(t)
        else:
            return self.false_event.execute(t)