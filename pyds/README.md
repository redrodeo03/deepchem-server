# DeepChem Data Science Client

A Python client package for interacting with the DeepChem server API. This package provides an easy-to-use interface for managing settings, uploading data, and submitting primitive jobs like featurization.

## Features

- **Settings Management**: Persistent configuration for profile, project, and server settings
- **Modular Architecture**: Base classes with inheritance for clean code organization
- **API Client**: Easy-to-use clients for interacting with DeepChem server endpoints
- **Data Upload**: Upload files to the DeepChem datastore
- **Featurization**: Submit featurization jobs with various featurizers
- **Health Checks**: Monitor server status

## Installation

### From Source (Development)

1. Clone the repository and navigate to the pyds directory
2. Install in editable mode:
```bash
pip install -e .
```

### For Development
Install with development dependencies:
```bash
pip install -e ".[dev]"
# or
pip install -r requirements-dev.txt
```

### Usage
After installation, import the package in your Python code:
```python
from pyds import Settings, Primitives, Data, BaseClient
```

## Quick Start

### 1. Configure Settings

```python
from pyds import Settings

# Initialize settings
settings = Settings()

# Configure your profile and project
settings.set_profile("my_profile")
settings.set_project("my_project")
settings.set_base_url("http://localhost:8000")  # Optional, defaults to localhost:8000

print(settings)  # Settings(profile='my_profile', project='my_project', base_url='http://localhost:8000')
```

Settings are automatically saved to `.pyds.settings.json` in the current directory.

### 2. Initialize API Clients

```python
from pyds import Primitives, Data

# Initialize clients with your settings
primitives_client = Primitives(settings)
data_client = Data(settings)
```

### 3. Upload Data (using Data)

```python
# Upload a file to the datastore using Data client
response = data_client.upload_data(
    file_path="path/to/your/data.csv",
    description="Sample dataset for analysis"
)

dataset_address = response['dataset_address']
print(f"Dataset uploaded to: {dataset_address}")
```

### 4. Submit Featurization Jobs (using Primitives)

```python
# Submit a featurization job using Primitives client
response = primitives_client.featurize(
    dataset_address=dataset_address,
    featurizer="ECFP",
    output="featurized_dataset",
    dataset_column="smiles",
    feat_kwargs={"radius": 2, "size": 1024},
    label_column="target"  # Optional
)

featurized_address = response['featurized_file_address']
print(f"Featurized dataset: {featurized_address}")
```

### 5. Train a Model

```python
# Submit a training job
response = primitives_client.train(
    dataset_address=featurized_address,
    model_type="random_forest_regressor",
    model_name="my_trained_model",
    init_kwargs={"n_estimators": 100},
    train_kwargs={"nb_epoch": 10}
)

model_address = response['trained_model_address']
print(f"Trained model: {model_address}")
```

### 6. Evaluate a Model

```python
# Submit an evaluation job
response = primitives_client.evaluate(
    dataset_addresses=[featurized_address],
    model_address=model_address,
    metrics=["roc_auc_score", "accuracy_score"],
    output_key="evaluation_results"
)

eval_address = response['evaluation_result_address']
print(f"Evaluation results: {eval_address}")
```

### 7. Run Inference

```python
# Submit an inference job
response = primitives_client.infer(
    model_address=model_address,
    data_address=dataset_address,
    output="inference_results",
    dataset_column="smiles",
    threshold=0.5  # For classification tasks
)

inference_address = response['inference_results_address']
print(f"Inference results: {inference_address}")
```

### 8. Health Check

```python
# Check server health
health_status = primitives_client.healthcheck()
print(f"Server status: {health_status}")
```

## Architecture

The package follows a clean inheritance structure:

```
BaseClient (base functionality)
├── Primitives (computation tasks)
└── Data (data operations)
```

- **BaseClient**: Contains all common functionality like HTTP requests, configuration validation, and shared utilities
- **Primitives**: Inherits from BaseClient, adds computation-specific methods like `featurize()`
- **Data**: Inherits from BaseClient, adds data management methods like `upload_data()`

This architecture eliminates code duplication and provides a consistent interface across all clients.

## API Reference

### BaseClient Class

The `BaseClient` is the foundation class that provides common functionality for all API clients:

- `__init__(settings, base_url)`: Initialize with settings and optional base URL override
- `_make_request(method, endpoint, **kwargs)`: Make HTTP requests to the API
- `_check_configuration()`: Validate that profile and project are configured
- `_get_profile_and_project(profile_name, project_name)`: Get profile/project with validation
- `healthcheck()`: Check server health status
- `get_settings()`: Get the current settings instance
- `get_base_url()`: Get the current base URL
- `set_base_url(url)`: Set a new base URL for this client instance
- `close()`: Close the HTTP session

### Settings Class

The `Settings` class manages configuration persistence:

- `set_profile(profile_name)`: Set the profile name
- `set_project(project_name)`: Set the project name  
- `set_base_url(url)`: Set the API base URL
- `get_profile()`: Get current profile name
- `get_project()`: Get current project name
- `get_base_url()`: Get current base URL
- `is_configured()`: Check if both profile and project are set
- `save()`: Save settings to JSON file
- `load()`: Load settings from JSON file
- `reset()`: Reset all settings and remove JSON file

### Primitives Class

The `Primitives` (inherits from `BaseClient`) provides methods for computation primitives:

- `featurize(dataset_address, featurizer, output, dataset_column, feat_kwargs=None, label_column=None)`: Submit featurization job
- `train(dataset_address, model_type, model_name, init_kwargs=None, train_kwargs=None)`: Submit model training job
- `evaluate(dataset_addresses, model_address, metrics, output_key, is_metric_plots=False)`: Submit model evaluation job
- `infer(model_address, data_address, output, dataset_column=None, shard_size=8192, threshold=None)`: Submit inference job
- All methods from `BaseClient` (healthcheck, configuration management, etc.)

### Data Class

The `Data` (inherits from `BaseClient`) provides methods for data management operations:

- `upload_data(file_path, filename=None, description=None, backend='local')`: Upload data file
- All methods from `BaseClient` (healthcheck, configuration management, etc.)

**Note**: All data operations (upload, download, delete) should be performed using `Data`, while computation tasks (featurize, train, evaluate, infer) should use `Primitives`.

## Available Endpoints

The package interfaces with the following DeepChem server endpoints:

### Primitive Endpoints
- `POST /primitive/featurize`: Submit featurization jobs
- `POST /primitive/train`: Submit model training jobs
- `POST /primitive/evaluate`: Submit model evaluation jobs
- `POST /primitive/infer`: Submit inference jobs

### Data Endpoints
- `POST /data/uploaddata`: Upload data to datastore

### System Endpoints
- `GET /healthcheck`: Check server health

## Configuration File

Settings are stored in `.pyds.settings.json` with the following structure:

```json
{
  "profile": "my_profile",
  "project": "my_project", 
  "base_url": "http://localhost:8000",
  "additional_settings": {
    "custom_key": "custom_value"
  }
}
```

## Example Usage

The usage examples above demonstrate how to use the package effectively.

## Error Handling

The clients include proper error handling:

- **Configuration Errors**: Raised when required settings (profile/project) are missing
- **File Errors**: Raised when upload files don't exist
- **API Errors**: Raised when HTTP requests fail or return errors

## Requirements

- Python 3.7+
- requests >= 2.28.0
- requests-toolbelt >= 0.10.0

## Development

To run tests:
```bash
pytest
``` 