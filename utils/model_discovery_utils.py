"""
Model Discovery Utilities for NFC Detector

This module provides a comprehensive framework for model discovery and organization, including:
- Dynamic discovery of model modules in the utils/models package
- Automatic importing of model test functions and metadata
- Model categorization system based on numeric prefixes (00-09: QC & Prep, 10-19: Front/Back Detection, etc.)
- Structured model information extraction and organization
- Tabular display of models with training status and performance metrics
- Command-line interface support for model selection and execution
- Integration with model metadata and performance tracking

These utilities support the image classification pipeline by providing a centralized
system for managing multiple model variants and their lifecycle states.
"""

import importlib
import pkgutil
import utils.models
from pathlib import Path
from tabulate import tabulate

from utils.model_metadata_utils import load_model_metadata
from utils.model_performance_utils import format_metrics_with_emoji

def determine_model_category(model_number):
    """Determine the model category based on model number"""
    try:
        # Extract first two digits and convert to integer for range comparison
        prefix = int(model_number[:2]) if len(model_number) >= 2 else -1
        
        # Categorize based on ranges
        if 0 <= prefix <= 9:  # 00-09 range
            return "QC & Prep"
        elif 10 <= prefix <= 19:  # 10-19 range
            return "Front/Back Detection"
        elif 30 <= prefix <= 39:  # 30-39 range
            return "Cut Classification"
        else:
            return "Other"
    except ValueError:
        # Handle case where model_number doesn't start with digits
        return "Other"

def discover_models():
    """
    Discover all model modules in the utils/models package and extract their metadata
    
    Returns:
        tuple: (models, model_choices) - dictionary of model info and list of CLI-friendly model names
    """
    models = {}
    model_choices = []
    
    # Find all model modules in the utils.models package
    for _, name, _ in pkgutil.iter_modules(utils.models.__path__):
        if name.startswith('m') and '_' in name:
            try:
                # Import the module
                module = importlib.import_module(f'utils.models.{name}')
                
                # Extract metadata
                model_name = getattr(module, 'MODEL_NAME', name)
                model_number = getattr(module, 'MODEL_NUMBER', '00')
                model_description = getattr(module, 'MODEL_DESCRIPTION', 'No description available')
                
                # Get category based on model number
                model_category = getattr(module, 'MODEL_CATEGORY', 
                                        determine_model_category(model_number))
                
                # Get the main test function
                # Assuming the function is named test_X where X is the model_name
                test_func = getattr(module, f'test_{model_name}', None)
                
                if test_func:
                    # Convert to command-line friendly format (for choices)
                    cli_name = model_name.replace('_', '-')
                    model_choices.append(cli_name)
                    
                    # Store model info
                    models[cli_name] = {
                        'module': module,
                        'function': test_func,
                        'name': model_name,
                        'number': model_number,
                        'description': model_description,
                        'category': model_category,
                        'filename': f"{model_number}_{model_name}_model.pkl"
                    }
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load model from {name}: {e}")
    
    return models, sorted(model_choices)

def list_models(models_path, available_models):
    """
    List all available models with their metadata in a formatted table
    
    Args:
        models_path: Path to models directory
        available_models: Dictionary of model info from discover_models()
    """
    print("\nAvailable Models:")
    print("================")
    
    # Table headers
    headers = ["Model Name", "Number", "Category", "Status", "CLI Parameter", "Metrics"]
    table_data = []
    
    # Sort models by their number for consistent display
    for cli_param, model_info in sorted(available_models.items(), key=lambda x: x[1]['number']):
        display_name = model_info['name'].replace('_', ' ').title()
        model_number = model_info['number']
        model_path = models_path / f"{model_number}_{model_info['name']}_model.pkl"
        
        # Determine category based on model number
        category = determine_model_category(model_number)
        
        # Check if model exists and get status
        if model_path.exists():
            status = "✓ Trained"
        else:
            status = "✗ Not trained"
        
        # Default empty metrics display
        metrics_display = ""
        
        # Get metrics from model metadata if available
        if model_path.exists():
            metadata = load_model_metadata(model_path)
            # Use the new function to get metrics with emoji
            metrics_display = format_metrics_with_emoji(metadata)
        
        # Add row to table data with CLI parameter
        table_data.append([display_name, model_number, category, status, cli_param, metrics_display])
    
    # Display the table using tabulate
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print("\nTo train a specific model: python 01_run_image_tests.py --only MODEL-NAME")
    print("For more information, run: python 01_run_image_tests.py -h")
