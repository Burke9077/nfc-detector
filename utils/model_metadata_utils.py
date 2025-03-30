"""
Model Metadata Utilities

This module provides functions for saving, loading, and comparing model metadata.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

def save_model_metadata(model_path: Path, metrics: Dict[str, float], overwrite: bool = True) -> bool:
    """
    Save model training metrics and timestamp to a metadata file
    
    Args:
        model_path: Path to the model file
        metrics: Dictionary of metric names and values
        overwrite: Whether to overwrite existing metadata
        
    Returns:
        bool: True if metadata was saved successfully
    """
    metadata_path = model_path.with_suffix('.metadata.json')
    
    # Don't overwrite existing metadata unless specified
    if metadata_path.exists() and not overwrite:
        return False
        
    # Add timestamp
    metadata = {
        "timestamp": time.time(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics
    }
    
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return False

def load_model_metadata(model_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load model metadata from a file
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary containing model metadata or None if not found
    """
    metadata_path = model_path.with_suffix('.metadata.json')
    
    if not metadata_path.exists():
        return None
        
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

def is_model_better(new_metrics: Dict[str, float], old_metrics: Dict[str, float]) -> bool:
    """
    Compare model metrics to determine if new model is better
    
    Args:
        new_metrics: Metrics from the new model
        old_metrics: Metrics from the old model
        
    Returns:
        True if the new model is better
    """
    # Priority of metrics (higher priority first)
    metric_priority = ['valid_accuracy', 'accuracy', 'valid_error_rate', 'error_rate', 
                       'valid_loss', 'train_loss']
    
    for metric in metric_priority:
        # Skip metrics that don't exist in both
        if metric not in new_metrics or metric not in old_metrics:
            continue
            
        # For accuracy metrics, higher is better
        if 'accuracy' in metric:
            if new_metrics[metric] > old_metrics[metric]:
                return True
            elif new_metrics[metric] < old_metrics[metric]:
                return False
        # For error and loss metrics, lower is better
        else:
            if new_metrics[metric] < old_metrics[metric]:
                return True
            elif new_metrics[metric] > old_metrics[metric]:
                return False
    
    # If we get here, models are equivalent or incomparable
    return False

def format_metadata_for_display(metadata: Dict[str, Any]) -> str:
    """
    Format model metadata for display in CLI output
    
    Args:
        metadata: Model metadata dictionary
        
    Returns:
        Formatted string with key metrics
    """
    if not metadata:
        return "No metadata available"
        
    metrics = metadata.get('metrics', {})
    created = metadata.get('created', 'Unknown')
    
    # Format key metrics, prioritizing validation metrics
    metric_display = []
    
    for metric_name in ['valid_accuracy', 'accuracy', 'valid_error_rate', 'error_rate', 'valid_loss']:
        if metric_name in metrics:
            # Format name for display
            display_name = metric_name.replace('valid_', 'val_')
            display_name = display_name.replace('_', ' ').title()
            
            # Format value based on metric type
            value = metrics[metric_name]
            if 'accuracy' in metric_name or 'error' in metric_name:
                formatted_value = f"{value:.2%}"  # Format as percentage
            else:
                formatted_value = f"{value:.4f}"
                
            metric_display.append(f"{display_name}: {formatted_value}")
    
    # Combine the most important metrics (limit to 3)
    metrics_str = ", ".join(metric_display[:3])
    
    return f"{metrics_str} (Created: {created})"
