# Standard models
import numpy as np
import pandas as pd

from time_blender.coordination_events import PastEvent, CumulativeEvent, ParentValueEvent, TemporarySwitch, Choice, \
    SeasonalEvent
from time_blender.core import LambdaEvent, ConstantEvent, wrapped_constant_param, Event, Invariant
from time_blender.deterministic_events import WaveEvent, ClockEvent, WalkEvent, IdentityEvent, ClipEvent
from time_blender.random_events import NormalEvent, wrap_in_resistance, BernoulliEvent, PoissonEvent

from clize import Parameter


from time_blender.util import shift_weekend_and_holidays
from time_blender.cli import cli_model, a_list


class SimpleModels:
    @staticmethod
    @cli_model
    def cycle(base:float=10.0, period:float=72, growth_rate:float=2):

        period_fluctuation = WalkEvent(NormalEvent(0, 1), initial_pos=period, capture_parent_value=False)
        amplitude_trend = ClockEvent() * ConstantEvent(base) * NormalEvent(3, 1)
        we = WaveEvent(period_fluctuation, amplitude_trend)

        capacity_trend = ClockEvent() * ConstantEvent(growth_rate*base) * NormalEvent(3, 0.1)

        return we + capacity_trend


class ClassicModels:

    @staticmethod
    @cli_model
    def ar(p: int, *, constant: float=0, error_mean: float=0, error_std: float=1, coefs_low: float=-1, coefs_high: float=1,
           coefs: a_list=None, error_event: Parameter.IGNORE=None, capture_parent_value=True):
        """
        Creates a new AR (autoreressive) model. The model's coeficients can either be generated automatically
        by providing coefs_low and coefs_high parameters, or be explicitly defined by providing a list in the
        coefs parameter.

        :param p: The order of the AR model (how far back should it look).
        :param constant: The model's constant term.
        :param error_mean: The mean of the normal error component.
        :param error_std: The standard deviation of the normal error component.
        :param coefs_low: If specified, defines the lower bound of the coeficients to be generated. If left None,
                          then the coefs parameter must be specified.
        :param coefs_high: If specified, defines the upper bound of the coeficients to be generated. If left None,
                          then the coefs parameter must be specified.
        :param coefs: A list or dict with numeric keys of the coeficients to be employed. Must have size p. The i-th
                      element (if list) or key (if dict) correspond to the i-th coeficient.
                      If this is specified, coefs_low and coefs_high are ignored.
        :param error_event: An error event. If specified, it is used instead of error_mean and error_std.
        :capture_parent_value: Whether the parent value should be used as the new current value to which
                               the event's execution is added. This is useful to embed the present event
                               into larger contexts and accumulate on top of their feedback.
        :return: An AR model.
        """

        # check coeficients
        if (coefs is None) and (coefs_low is None) and (coefs_high is None):
            raise ValueError("Either coefs or coefs_loe, coefs_high must be specified.")

        # check error events
        if (coefs is error_mean) and (error_std is None) and (error_event is None):
            raise ValueError("Some error must be specified.")

        # Start with the model's constant.
        if error_event is not None:
            if not isinstance(constant, Event):
                x = ConstantEvent(constant, parallel_events=[error_event])
            else:
                x = constant.parallel_to(error_event)
        else:
            if not isinstance(constant, Event):
                x = ConstantEvent(constant)
            else:
                x = constant

        past = []

        # Add the autoregressive terms
        for i in range(0, p):

            if coefs is not None:
                alpha = coefs[i]
            else:
                alpha = np.random.uniform(coefs_low, coefs_high)

            pe = PastEvent(i + 1, allow_learning=False)
            past.append(pe)

            if error_event is not None:
                error = PastEvent(i, allow_learning=False)
                error.refers_to(error_event)
            else:
                error = NormalEvent(error_mean, error_std)

            x = x + pe * ConstantEvent(alpha) + error

        # connect past events to the series to which they refer to
        for pe in past:
            if capture_parent_value:
                pe.refers_to(ParentValueEvent(x))

            else:
                pe.refers_to(x)

        return x

    @staticmethod
    @cli_model
    def ma(q, *, series_mean: float=0, error_mean: float=0, error_std: float=1,
           coefs_low: float=-1, coefs_high: float=1, coefs: a_list=None, error_event: Parameter.IGNORE=None):
        """
        Creates a new MA (Moving Average) model. The model's coeficients can either be generated automatically
        by providing coefs_low and coefs_high parameters (default), or be explicitly defined by providing a list in the
        coefs parameter.

        :param q: The order of the MA model (how far back should it look).
        :param series_mean: The mean of the series.
        :param error_mean: The mean of the normal error of each past random shock.
        :param error_std: The standard deviation of the normal error of each past random shock.
        :param coefs_low: If specified, defines the lower bound of the coeficients to be generated. If left None,
                          then the coefs parameter must be specified.
        :param coefs_high: If specified, defines the upper bound of the coeficients to be generated. If left None,
                          then the coefs parameter must be specified.
        :param coefs: A list or dict with numeric keys of the coeficients to be employed. Must have size p. The i-th
                      element (if list) or key (if dict) correspond to the i-th coeficient.
                      If this is specified, coefs_low and coefs_high are ignored.
        :param error_event: An error event. If specified, it is used instead of error_mean and error_std.

        :return: The MA model.
        """

        # check coeficients
        if (coefs is None) and (coefs_low is None) and (coefs_high is None):
            raise ValueError("Either coefs or coefs_loe, coefs_high must be specified.")

        # error shocks
        if error_event is None:
            error_event = NormalEvent(error_mean, error_std)

        # Put the mean term first
        x = ConstantEvent(series_mean, parallel_events=[error_event])

        past = []

        # Add model terms
        for i in range(0, q):

            if coefs is not None:
                alpha = coefs[i]
            else:
                alpha = np.random.uniform(coefs_low, coefs_high)

            p = PastEvent(i + 1, allow_learning=False)
            past.append(p)
            x = x + p * ConstantEvent(alpha)

        # connect past events to the series to which they refer to
        for p in past:
            p.refers_to(error_event)

        return x

    @staticmethod
    @cli_model
    def arma(p, q, constant: float=0, error_mean: float=0, error_std: float=1,
             ar_coefs_low: float=-1, ar_coefs_high: float=1, ar_coefs: a_list=None,
             ma_coefs_low: float=-1, ma_coefs_high: float=1, ma_coefs: a_list=None,
             capture_parent_value=True):
        """
        Creates an ARMA model. This differs slightly from simply summing AR and MA models, because here a common
        normal error series is also provided

        :param p:
        :param q:
        :param constant:
        :param error_mean:
        :param error_std:
        :param ar_coefs_low:
        :param ar_coefs_high:
        :param ar_coefs:
        :param ma_coefs_low:
        :param ma_coefs_high:
        :param ma_coefs:
        :param capture_parent_value:
        :return:
        """

        # common error series
        error_event = NormalEvent(error_mean, error_std)

        m1 = ClassicModels.ar(p, constant=constant, coefs_low=ar_coefs_low, coefs_high=ar_coefs_high, coefs=ar_coefs,
                error_event=error_event, capture_parent_value=capture_parent_value)

        m2 = ClassicModels.ma(q, series_mean=0.0, coefs_low=ma_coefs_low, coefs_high=ma_coefs_high, coefs=ma_coefs,
                error_event=error_event)

        return m1 + m2

    @staticmethod
    @cli_model
    def arima(p, q, constant: float=0, error_mean: float=0, error_std: float=1,
             ar_coefs_low: float=-1, ar_coefs_high: float=1, ar_coefs: a_list=None,
             ma_coefs_low: float=-1, ma_coefs_high: float=1, ma_coefs: a_list=None,
             capture_parent_value=True):
        """
        Creates an ARIMA model. This adds the integration not found in ARMA. That is to say, values are accumulated
        over time.

        :param p:
        :param q:
        :param constant:
        :param error_mean:
        :param error_std:
        :param ar_coefs_low:
        :param ar_coefs_high:
        :param ar_coefs:
        :param ma_coefs_low:
        :param ma_coefs_high:
        :param ma_coefs:
        :param capture_parent_value:
        :return:
        """

        return CumulativeEvent(\
                    ClassicModels.arma(p=p, q=1, constant=constant, error_mean=error_mean, error_std=error_std,
                                       ar_coefs_low=ar_coefs_low, ar_coefs_high=ar_coefs_high, ar_coefs=ar_coefs,
                                       ma_coefs_low=ma_coefs_low, ma_coefs_high=ma_coefs_high, ma_coefs=ma_coefs,
                                       capture_parent_value=capture_parent_value))


class CompositeModelBuilder:
    """
    A builder class for creating composite models by combining multiple components.
    This provides a flexible way to construct complex time series models.
    """
    
    def __init__(self):
        self.components = []
        self.weights = []
        self.operations = []
        
    def add_component(self, component: Event, weight: float = 1.0, operation: str = "add"):
        """
        Add a component to the composite model.
        
        :param component: The event component to add
        :param weight: Weight for the component
        :param operation: Operation to apply - "add", "multiply", "subtract", "divide"
        """
        self.components.append(component)
        self.weights.append(weight)
        self.operations.append(operation)
        return self
        
    def build(self) -> Event:
        """
        Build and return the composite model.
        """
        if not self.components:
            return ConstantEvent(0.0)
            
        # Start with the first component
        if self.weights[0] != 1.0:
            result = self.components[0] * ConstantEvent(self.weights[0])
        else:
            result = self.components[0]
            
        # Apply remaining components
        for i in range(1, len(self.components)):
            component = self.components[i]
            weight = self.weights[i]
            operation = self.operations[i]
            
            # Apply weight
            if weight != 1.0:
                weighted_component = component * ConstantEvent(weight)
            else:
                weighted_component = component
                
            # Apply operation
            if operation == "add":
                result = result + weighted_component
            elif operation == "multiply":
                result = result * weighted_component
            elif operation == "subtract":
                result = result - weighted_component
            elif operation == "divide":
                result = result / weighted_component
                
        return result


class ModelLibrary:
    """
    A library of pre-built model templates that can be easily customized.
    """
    
    @staticmethod
    def trend_model(trend_type: str = "linear", slope: float = 0.1, 
                   acceleration: float = 0.0, initial_value: float = 0.0) -> Event:
        """
        Create a trend model.
        
        :param trend_type: Type of trend - "linear", "exponential", "quadratic"
        :param slope: Slope parameter
        :param acceleration: Acceleration parameter (for quadratic)
        :param initial_value: Initial value
        """
        if trend_type == "linear":
            return ConstantEvent(initial_value) + WalkEvent(ConstantEvent(slope), initial_pos=0.0)
        elif trend_type == "exponential":
            def exp_trend(t, i, memory, sub_events):
                return initial_value * (1 + slope) ** (i / 365.0)
            return LambdaEvent(exp_trend, sub_events={})
        elif trend_type == "quadratic":
            def quad_trend(t, i, memory, sub_events):
                return initial_value + slope * i + 0.5 * acceleration * i ** 2
            return LambdaEvent(quad_trend, sub_events={})
        else:
            return ConstantEvent(initial_value)
            
    @staticmethod
    def seasonal_model(periods: list, amplitudes: list, phases: list = None) -> Event:
        """
        Create a seasonal model with multiple seasonal components.
        
        :param periods: List of seasonal periods
        :param amplitudes: List of seasonal amplitudes
        :param phases: List of seasonal phases
        """
        if phases is None:
            phases = [0.0] * len(periods)
            
        components = []
        for period, amplitude, phase in zip(periods, amplitudes, phases):
            components.append(WaveEvent(period, amplitude, phase=phase))
            
        if not components:
            return ConstantEvent(0.0)
        elif len(components) == 1:
            return components[0]
        else:
            result = components[0]
            for component in components[1:]:
                result = result + component
            return result
            
    @staticmethod
    def noise_model(noise_type: str = "normal", mean: float = 0.0, 
                   std: float = 1.0) -> Event:
        """
        Create a noise model.
        
        :param noise_type: Type of noise - "normal", "uniform"
        :param mean: Mean of the noise
        :param std: Standard deviation of the noise
        """
        if noise_type == "normal":
            return NormalEvent(mean, std)
        elif noise_type == "uniform":
            def uniform_noise(t, i, memory, sub_events):
                return np.random.uniform(mean - std, mean + std)
            return LambdaEvent(uniform_noise, sub_events={})
        else:
            return NormalEvent(mean, std)
            
    @staticmethod
    def regime_switching_model(regimes: list, transitions: list = None) -> Event:
        """
        Create a regime switching model.
        
        :param regimes: List of regime events
        :param transitions: Transition probability matrix
        """
        from time_blender.extended_events import RegimeSwitchingEvent
        return RegimeSwitchingEvent(regimes, transitions)
        
    @staticmethod
    def stochastic_process_model(process_type: str = "ou", **kwargs) -> Event:
        """
        Create a stochastic process model.
        
        :param process_type: Type of process - "ou", "gbm", "jump"
        :param kwargs: Process parameters
        """
        from time_blender.extended_events import StochasticProcessEvent
        return StochasticProcessEvent(process_type=process_type, **kwargs)


class ModelComposer:
    """
    A high-level composer for creating complex time series models by combining templates.
    """
    
    @staticmethod
    def create_custom_model(trend_params: dict = None, seasonal_params: dict = None,
                          noise_params: dict = None, regime_params: dict = None,
                          stochastic_params: dict = None) -> Event:
        """
        Create a custom model by combining different components.
        
        :param trend_params: Parameters for trend component
        :param seasonal_params: Parameters for seasonal component
        :param noise_params: Parameters for noise component
        :param regime_params: Parameters for regime switching component
        :param stochastic_params: Parameters for stochastic process component
        :return: Combined model event
        """
        components = []
        
        # Add trend component
        if trend_params:
            trend = ModelLibrary.trend_model(**trend_params)
            components.append(trend)
            
        # Add seasonal component
        if seasonal_params:
            seasonal = ModelLibrary.seasonal_model(**seasonal_params)
            components.append(seasonal)
            
        # Add noise component
        if noise_params:
            noise = ModelLibrary.noise_model(**noise_params)
            components.append(noise)
            
        # Add regime switching component
        if regime_params:
            regime = ModelLibrary.regime_switching_model(**regime_params)
            components.append(regime)
            
        # Add stochastic process component
        if stochastic_params:
            stochastic = ModelLibrary.stochastic_process_model(**stochastic_params)
            components.append(stochastic)
            
        # Combine all components
        if not components:
            return ConstantEvent(0.0)
        elif len(components) == 1:
            return components[0]
        else:
            result = components[0]
            for component in components[1:]:
                result = result + component
            return result