# MultiAgentTS: Multi-Agent Time Series Data Generation and Analysis System

A sophisticated multi-agent system for automated time series data generation and analysis using natural language control. The system leverages multiple AI agents working collaboratively to generate synthetic time series data with complex patterns and realistic characteristics.

## Key Features

- **Multi-Agent Architecture**: Multiple specialized agents (CodeAgent,ExpertAgent, AnalysisAgent, Discriminator) work collaboratively to automate the entire data generation pipeline.
- **Modular Design**: Clean separation of concerns with dedicated modules for code generation, analysis, and time series synthesis.
- **Flexible Time Series Generation**: The time_blender library provides rich event models and coordination mechanisms for generating complex temporal patterns.
- **Automated Iterative Refinement**: Multi-round iteration with automatic code generation, execution, analysis, and feedback-based optimization.
- **Comprehensive Analysis**: Statistical analysis, data validation, and quality assessment of generated data.


## Project Structure

```
.
├── Agents/                        # Multi-agent system module
│   ├── code_agent.py              # Code generation agent
│   ├── expert_agent.py            # Expert agent for description expansion
│   ├── code_generator.py          # Code generator implementation
│   ├── code_worker.py             # Code executor implementation
│   ├── Reflector.py               # Reflector for feedback
│   └── __init__.py                # Module initialization
├── Analysis_pipeline/             # Analysis pipeline module
│   ├── analysis_agent.py          # Analysis agent implementation
│   ├── CSV_statistical_feature_extraction.py  # Statistical feature extraction
│   ├── data_validation.py         # Data validation module
│   ├── test.py                    # Test file
│   └── __init__.py                # Module initialization
├── Data/                          # Generated data and utilities
│   └── plot_script.py             # Plotting script
├── Dataset/                       # Original datasets
│   ├── ETT/                       # Electricity Transformer Temperature data
│   ├── POWER/                     # Power consumption data
│   └── check_plot.py              # Data checking script
├── time_blender/                  # Time series generation core library
│   ├── core.py                    # Core implementation
│   ├── models.py                  # Base models
│   ├── extended_events.py         # Extended event models
│   ├── advanced_models.py         # Advanced model implementations
│   ├── deterministic_events.py    # Deterministic events
│   ├── random_events.py           # Random events
│   ├── coordination_events.py     # Coordination events
│   ├── util.py                    # Utility functions
│   ├── cli.py                     # Command-line interface
│   ├── config.py                  # Configuration file
│   └── __init__.py                # Module initialization
├── visualizations/                # Generated visualization outputs
├── config.py                      # Project configuration management
├── shared_utils.py                # Shared utility functions
├── extracted_script.py            # Data extraction script
├── run_ett.py                     # ETT dataset execution script
├── run_power.py                   # POWER dataset execution script
├── requirement.txt                # Project dependencies
├── .env.example                   # Environment variables template
└── README.md                      # Project documentation
```

## Installation

```bash
pip install -r requirement.txt
```

## Configuration

Before running the scripts, you need to configure the following environment variables:

### Method 1: Using .env file (Recommended)

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and fill in your API key and configuration:
   ```
   API_KEY=your-api-key-here
   BASE_URL=https://api.example.com/v1
   CODE_GENERATOR_MODEL=claude-sonnet-4-1-thinking
   ANALYSIS_MODEL=gpt-4o
   REFLECTOR_MODEL=claude-sonnet-4-1-thinking
   ```

### Method 2: Using system environment variables

```bash
export API_KEY="your-api-key"
export BASE_URL="https://api.example.com/v1"
export CODE_GENERATOR_MODEL="claude-sonnet-4-1-thinking"
export ANALYSIS_MODEL="gpt-4o"
export REFLECTOR_MODEL="claude-sonnet-4-1-thinking"
```

### Method 3: Setting in code (Not recommended for production)

Edit the default values in `config.py`.

## Usage

1. Configure environment variables (see Configuration section above)
2. Run the corresponding dataset script:
   ```bash
   python run_ecg.py     # Run ECG dataset
   python run_ett.py     # Run ETT dataset
   python run_power.py   # Run POWER dataset
   ```

## Key Improvements

### 1. Multi-Agent Collaboration Framework
- **CodeAgent**: Generates executable Python code for time series synthesis
- **ExpertAgent**: Enriches dataset descriptions with domain knowledge and contextual information
- **AnalysisAgent**: Performs comprehensive statistical analysis and feature extraction
- **Discriminator**: Evaluates data quality and provides feedback for iterative refinement

### 2. Unified Configuration Management
- Centralized API key and model configuration through config.py
- Support for environment variable configuration of sensitive information
- Provided `.env.example` template file for easy setup

### 3. Flexible Time Series Generation with time_blender
- Extended event models (extended_events.py) providing diverse event components
- Advanced model implementations for complex temporal patterns
- Support for regime switching, external factors, and stochastic processes
- Composable event system for creating realistic synthetic data

### 4. Comprehensive Data Analysis Pipeline
- Statistical feature extraction from time series data
- Data validation module for quality assessment
- Comparison between real and synthetic data distributions
- Automated analysis reports and visualizations

### 5. Iterative Refinement Process
- Multi-round iteration with automatic feedback loops
- Discriminator-based evaluation for quality improvement
- Support for natural language-driven customization
- Automated code generation and execution

### 6. Modular and Extensible Architecture
- Clean separation of concerns with dedicated modules
- Easy integration of new agents and event types
- Shared utilities for common operations
- Enhanced error handling and input validation

## Extended Event Models in time_blender

The time_blender library includes the following extended event types that can be freely combined to generate various complex time series:

1. **RegimeSwitchingEvent** - Regime switching events that simulate different behaviors under different states
2. **ExternalFactorEvent** - External factor events that simulate the impact of external variables on time series
3. **ComplexWaveEvent** - Complex wave events that can combine multiple waveforms
4. **StochasticProcessEvent** - Stochastic process events supporting OU processes, geometric Brownian motion, etc.
5. **RegimeDependentEvent** - Regime-dependent events where behavior changes with state
6. **NonlinearTransformationEvent** - Nonlinear transformation events that apply nonlinear transformations to other events
7. **ConditionalEvent** - Conditional events that select different behaviors based on conditions

These events can be freely combined to create complex models capable of simulating various real-world time series.