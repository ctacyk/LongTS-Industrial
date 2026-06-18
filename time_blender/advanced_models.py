"""
Advanced time series models for the time_blender library.
This module contains more sophisticated models that can capture complex patterns in time series data.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union

from time_blender.core import Event, LambdaEvent, ConstantEvent
from time_blender.models import ClassicModels, BankingModels, EconomicModels, EcologyModels
from time_blender.random_events import NormalEvent, BernoulliEvent, PoissonEvent
from time_blender.deterministic_events import WaveEvent, WalkEvent, ClipEvent
from time_blender.coordination_events import SeasonalEvent, Choice, CumulativeEvent, PastEvent
from time_blender.util import shift_weekend_and_holidays


class AdvancedModels:
    """Advanced time series models with complex dependencies and multivariate relationships."""
    
    @staticmethod
    def multivariate_ecg_model(
        heart_rate_base: float = 70,
        heart_rate_variability: float = 10,
        signal1_noise_level: float = 0.5,
        signal2_noise_level: float = 0.5,
        correlation_strength: float = 0.8
    ) -> Tuple[Event, Event]:
        """
        Advanced ECG model that generates two correlated signals representing different ECG leads.
        
        :param heart_rate_base: Base heart rate in beats per minute
        :param heart_rate_variability: Variability in heart rate
        :param signal1_noise_level: Noise level for first signal
        :param signal2_noise_level: Noise level for second signal
        :param correlation_strength: Strength of correlation between signals
        :return: Tuple of two events representing the two ECG signals
        """
        
        # Convert heart rate to period in time steps (assuming 128 Hz sampling as in the dataset)
        sampling_rate = 128  # samples per second
        heart_period_base = ConstantEvent(sampling_rate * 60 / heart_rate_base)
        heart_period_variability = NormalEvent(0, heart_rate_variability * 0.1)
        
        # Heartbeat pattern - more realistic R-peak shape
        def heartbeat_pattern(t, i, memory, sub_events):
            # Time since last heartbeat
            if 'last_beat' not in memory:
                memory['last_beat'] = i
                
            time_since_beat = i - memory['last_beat']
            
            # Heart period with variability
            period = sub_events['period_base'].execute(t) + sub_events['period_var'].execute(t)
            
            if time_since_beat >= period:
                memory['last_beat'] = i
                time_since_beat = 0
                
            # Normalized time within heartbeat cycle
            norm_time = time_since_beat / period if period > 0 else 0
            
            # More realistic ECG waveform with distinct P, QRS, and T waves
            if norm_time < 0.1:  # P wave
                value = np.sin(2 * np.pi * norm_time * 5)
            elif norm_time < 0.2:  # PR segment
                value = 0.1 * (0.2 - norm_time)
            elif norm_time < 0.25:  # Q wave
                value = -0.3 * np.sin(2 * np.pi * (norm_time - 0.2) * 10)
            elif norm_time < 0.35:  # R wave (main peak)
                value = np.sin(np.pi * (norm_time - 0.25) * 10)
            elif norm_time < 0.45:  # S wave
                value = -0.2 * np.sin(np.pi * (norm_time - 0.35) * 5)
            elif norm_time < 0.6:  # ST segment
                value = 0.1 * (0.6 - norm_time)
            elif norm_time < 0.8:  # T wave
                value = 0.4 * np.sin(np.pi * (norm_time - 0.6) * 2.5)
            else:  # TP segment
                value = 0.05 * np.sin(2 * np.pi * (norm_time - 0.8) * 2)
                
            return value
        
        # Base heartbeat event
        heartbeat_base = LambdaEvent(
            heartbeat_pattern,
            sub_events={
                'period_base': heart_period_base,
                'period_var': heart_period_variability
            }
        )
        
        # Signal 1 with noise
        signal1_noise = NormalEvent(0, signal1_noise_level)
        signal1 = heartbeat_base + signal1_noise
        
        # Signal 2 with correlation to signal 1
        signal2_noise = NormalEvent(0, signal2_noise_level)
        correlation_factor = ConstantEvent(correlation_strength)
        signal2 = correlation_factor * heartbeat_base + signal2_noise
        
        return signal1, signal2

    @staticmethod
    def ett_model(
        base_temperature: float = 25.0,
        seasonal_amplitude: float = 10.0,
        daily_amplitude: float = 5.0,
        load_variability: float = 0.2,
        noise_level: float = 0.5
    ) -> Tuple[Event, Event, Event, Event, Event, Event, Event]:
        """
        Advanced ETT (Electricity Transformer Temperature) model with realistic load and temperature patterns.
        
        :param base_temperature: Base oil temperature
        :param seasonal_amplitude: Amplitude of seasonal temperature variation
        :param daily_amplitude: Amplitude of daily temperature variation
        :param load_variability: Variability in load patterns
        :param noise_level: General noise level
        :return: Tuple of events for OT, HUFL, HULL, MUFL, MULL, LUFL, LULL
        """
        
        # Time components
        seasonal_wave = WaveEvent(period=365, amplitude=seasonal_amplitude)
        daily_wave = WaveEvent(period=24, amplitude=daily_amplitude)
        
        # Base temperature with seasonal and daily patterns
        base_temp = ConstantEvent(base_temperature)
        temperature_trend = seasonal_wave + daily_wave
        temp_noise = NormalEvent(0, noise_level)
        oil_temperature = base_temp + temperature_trend + temp_noise
        
        # Load patterns with realistic correlations
        # High Useful Load - peaks during day
        hufl_base = WaveEvent(period=24, amplitude=ConstantEvent(50) * load_variability)
        hufl_trend = seasonal_wave * 0.5  # Seasonal variation
        hufl_noise = NormalEvent(0, 10 * load_variability)
        hufl = ClipEvent(ConstantEvent(100) + hufl_base + hufl_trend + hufl_noise, min_value=0)
        
        # High Useless Load - random spikes
        hull_spikes = PoissonEvent(0.1) * NormalEvent(20, 5)
        hull_noise = NormalEvent(0, 2 * load_variability)
        hull = ClipEvent(hull_spikes + hull_noise, min_value=0)
        
        # Middle Useful Load - moderate daily pattern
        mufl_base = WaveEvent(period=24, amplitude=ConstantEvent(30) * load_variability) * 0.7
        mufl_trend = seasonal_wave * 0.3
        mufl_noise = NormalEvent(0, 5 * load_variability)
        mufl = ClipEvent(ConstantEvent(50) + mufl_base + mufl_trend + mufl_noise, min_value=0)
        
        # Middle Useless Load - occasional inefficiencies
        mull_spikes = PoissonEvent(0.05) * NormalEvent(10, 3)
        mull_noise = NormalEvent(0, 1 * load_variability)
        mull = ClipEvent(mull_spikes + mull_noise, min_value=0)
        
        # Low Useful Load - night time base load
        lufl_base = WaveEvent(period=24, amplitude=ConstantEvent(20) * load_variability) * 0.5
        lufl_trend = seasonal_wave * 0.2
        lufl_noise = NormalEvent(0, 3 * load_variability)
        lufl = ClipEvent(ConstantEvent(30) + lufl_base + lufl_trend + lufl_noise, min_value=0)
        
        # Low Useless Load - minimal inefficiencies
        lull_spikes = PoissonEvent(0.02) * NormalEvent(5, 2)
        lull_noise = NormalEvent(0, 0.5 * load_variability)
        lull = ClipEvent(lull_spikes + lull_noise, min_value=0)
        
        return oil_temperature, hufl, hull, mufl, mull, lufl, lull

    @staticmethod
    def power_consumption_model(
        base_consumption: float = 1000,
        temperature_sensitivity: float = 10,
        humidity_sensitivity: float = 2,
        wind_sensitivity: float = -1,
        daily_pattern_strength: float = 0.5,
        seasonal_pattern_strength: float = 0.3,
        noise_level: float = 50
    ) -> Tuple[Event, Event, Event, Event, Event, Event]:
        """
        Advanced power consumption model for Tetouan city with weather dependencies.
        
        :param base_consumption: Base power consumption level
        :param temperature_sensitivity: Sensitivity to temperature changes
        :param humidity_sensitivity: Sensitivity to humidity changes
        :param wind_sensitivity: Sensitivity to wind speed (negative as wind cools)
        :param daily_pattern_strength: Strength of daily consumption patterns
        :param seasonal_pattern_strength: Strength of seasonal consumption patterns
        :param noise_level: General noise level
        :return: Tuple of events for Temperature, Humidity, Wind Speed, General Diffuse Flows, 
                Diffuse Flows, and Power Consumption
        """
        
        # Weather patterns
        temp_base = ConstantEvent(20)  # Base temperature in Celsius
        temp_seasonal = WaveEvent(period=365, amplitude=15)  # Seasonal variation
        temp_daily = WaveEvent(period=24, amplitude=10)  # Daily variation
        temp_noise = NormalEvent(0, 3)
        temperature = temp_base + temp_seasonal + temp_daily + temp_noise
        
        # Humidity - inversely related to temperature with daily pattern
        humidity_base = ConstantEvent(60)
        humidity_temp_effect = temperature * -0.5  # Cooler air holds less moisture
        humidity_daily = WaveEvent(period=24, amplitude=20)
        humidity_noise = NormalEvent(0, 5)
        humidity = ClipEvent(humidity_base + humidity_temp_effect + humidity_daily + humidity_noise, 
                            min_value=0, max_value=100)
        
        # Wind speed - random with seasonal pattern
        wind_base = ConstantEvent(3)
        wind_seasonal = WaveEvent(period=365, amplitude=2)
        wind_random = NormalEvent(0, 2)
        wind_speed = ClipEvent(wind_base + wind_seasonal + wind_random, min_value=0)
        
        # Diffuse flows - geological/geothermal factors
        diffuse_base = ConstantEvent(50)
        diffuse_variations = WaveEvent(period=30, amplitude=10)  # Monthly variations
        diffuse_noise = NormalEvent(0, 5)
        general_diffuse_flows = diffuse_base + diffuse_variations + diffuse_noise
        diffuse_flows = general_diffuse_flows * 0.8 + NormalEvent(0, 3)
        
        # Power consumption - complex function of weather and time patterns
        # Base consumption with daily and seasonal patterns
        consumption_base = ConstantEvent(base_consumption)
        consumption_daily = WaveEvent(period=24, amplitude=base_consumption * daily_pattern_strength)
        consumption_seasonal = WaveEvent(period=365, amplitude=base_consumption * seasonal_pattern_strength)
        
        # Weather effects on consumption
        temp_effect = temperature * temperature_sensitivity
        humidity_effect = humidity * humidity_sensitivity
        wind_effect = wind_speed * wind_sensitivity
        
        # Weekend effect - lower consumption
        weekend_effect = SeasonalEvent(
            event=ConstantEvent(-base_consumption * 0.2),
            is_weekend=True,
            fill_with_previous=False,
            default=0
        )
        
        # Noise
        consumption_noise = NormalEvent(0, noise_level)
        
        # Total consumption
        power_consumption = ClipEvent(
            consumption_base + consumption_daily + consumption_seasonal + 
            temp_effect + humidity_effect + wind_effect + 
            weekend_effect + consumption_noise,
            min_value=0
        )
        
        return temperature, humidity, wind_speed, general_diffuse_flows, diffuse_flows, power_consumption

    @staticmethod
    def correlated_multivariate_model(
        n_series: int = 3,
        base_values: Optional[List[float]] = None,
        correlations: Optional[np.ndarray] = None,
        noise_levels: Optional[List[float]] = None
    ) -> List[Event]:
        """
        Create a multivariate model with specified correlations between series.
        
        :param n_series: Number of time series to generate
        :param base_values: Base values for each series
        :param correlations: Correlation matrix (n_series x n_series)
        :param noise_levels: Noise level for each series
        :return: List of correlated events
        """
        
        if base_values is None:
            base_values = [0.0] * n_series
            
        if noise_levels is None:
            noise_levels = [1.0] * n_series
            
        if correlations is None:
            # Default to weak positive correlations
            correlations = np.full((n_series, n_series), 0.3)
            np.fill_diagonal(correlations, 1.0)
        
        # Generate independent noise components
        noises = [NormalEvent(0, noise_levels[i]) for i in range(n_series)]
        
        # Create correlated series using Cholesky decomposition
        L = np.linalg.cholesky(correlations)
        
        events = []
        for i in range(n_series):
            # Base value
            base = ConstantEvent(base_values[i])
            
            # Linear combination of noise components
            components = []
            for j in range(i+1):
                if L[i, j] != 0:
                    components.append(ConstantEvent(L[i, j]) * noises[j])
            
            # Sum all components
            if len(components) == 1:
                series = base + components[0]
            else:
                series = base
                for comp in components:
                    series = series + comp
                    
            events.append(series)
            
        return events