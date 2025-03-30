"""
Model Performance Utilities for NFC Detector

This module provides comprehensive utilities for evaluating and comparing model performance, including:
- Accuracy extraction and interpretation from model metadata
- Visual indicators (emojis) for quick assessment of model quality
- Quality categorization based on accuracy thresholds
- Performance change tracking and formatted reporting
- Model comparison logic with multi-metric evaluation
- Quality threshold-based decisions for model retraining

These utilities support the image classification pipeline by providing clear metrics
for model evaluation, enabling automated decision-making about model replacement,
and offering human-friendly performance reporting for command line interfaces.
"""

def get_best_accuracy_from_metadata(metadata):
    """Extract the best accuracy from model metadata"""
    if not metadata or 'metrics' not in metadata:
        return 0.0
    
    accuracy = metadata['metrics'].get('accuracy')
    if accuracy is None and 'tta_accuracy' in metadata['metrics']:
        accuracy = metadata['metrics'].get('tta_accuracy')
    
    try:
        return float(accuracy) if accuracy is not None else 0.0
    except (ValueError, TypeError):
        return 0.0

def get_accuracy_emoji(accuracy):
    """Return an emoji based on the accuracy value"""
    if accuracy is None:
        return ""
    
    try:
        accuracy = float(accuracy)
        if accuracy > 0.999:
            return "✅"  # Really happy face for excellent accuracy
        elif accuracy > 0.97:
            return "😄"  # Slightly smiling face for good accuracy
        elif accuracy > 0.90:
            return "😐"  # Straight face for acceptable accuracy
        elif accuracy > 0.80:
            return "🙁"  # Frowny face for mediocre accuracy
        else:
            return "😭"  # Crying face for poor accuracy
    except (ValueError, TypeError):
        return ""

def get_model_quality_category(accuracy):
    """Return quality category based on accuracy value"""
    if accuracy is None:
        return None
    
    try:
        accuracy = float(accuracy)
        if accuracy > 0.999:
            return "excellent" 
        elif accuracy > 0.97:
            return "good"      
        elif accuracy > 0.90:
            return "acceptable"
        elif accuracy > 0.80:
            return "mediocre" 
        else:
            return "poor"   
    except (ValueError, TypeError):
        return None

def format_performance_change(change, percent_change):
    """Format the accuracy change with appropriate sign and color indicators"""
    if change is None:
        return "N/A"
    
    change_sign = '+' if change >= 0 else ''
    return f"{change_sign}{change:.4f} ({change_sign}{percent_change:.2f}%)"

def compare_model_performance(new_model_metadata, existing_model_path, model_name, model_performance_tracker=None):
    """
    Compare the performance of a newly trained model with an existing one.
    Returns True if new model should replace the existing one.
    Also tracks performance changes for later reporting.
    
    Args:
        new_model_metadata: Metadata dictionary for the new model
        existing_model_path: Path object to the existing model
        model_name: Name of the model for tracking purposes
        model_performance_tracker: Optional dictionary to track performance changes
        
    Returns:
        bool: True if the new model should replace the existing one
    """
    # If no tracker provided, create a local one
    tracker_to_use = model_performance_tracker or {}
    
    # If existing model doesn't exist, always save new model
    if not existing_model_path.exists():
        # Store info about new model (no comparison available)
        tracker_to_use[model_name] = {
            'new_accuracy': get_best_accuracy_from_metadata(new_model_metadata),
            'old_accuracy': None,
            'change': None,
            'percent_change': None,
            'is_improvement': True  # New model is always an improvement if no previous model
        }
        return True
    
    # Load existing model metadata
    from utils.model_metadata_utils import load_model_metadata
    existing_metadata = load_model_metadata(existing_model_path)
    
    # If existing metadata is invalid or missing metrics, replace it
    if not existing_metadata or 'metrics' not in existing_metadata:
        tracker_to_use[model_name] = {
            'new_accuracy': get_best_accuracy_from_metadata(new_model_metadata),
            'old_accuracy': None,
            'change': None,
            'percent_change': None,
            'is_improvement': True
        }
        return True
    
    # If new metadata is invalid, don't replace existing model
    if not new_model_metadata or 'metrics' not in new_model_metadata:
        print("Warning: New model has no metrics. Keeping existing model.")
        return False
    
    # Get accuracy metrics from both models
    new_accuracy = get_best_accuracy_from_metadata(new_model_metadata)
    existing_accuracy = get_best_accuracy_from_metadata(existing_metadata)
    
    # Calculate accuracy change
    accuracy_change = new_accuracy - existing_accuracy
    percent_change = (accuracy_change / existing_accuracy) * 100 if existing_accuracy > 0 else 0
    
    # Store performance comparison info
    tracker_to_use[model_name] = {
        'new_accuracy': new_accuracy,
        'old_accuracy': existing_accuracy,
        'change': accuracy_change,
        'percent_change': percent_change,
        'is_improvement': new_accuracy > existing_accuracy
    }
    
    # Compare accuracies first
    if new_accuracy > existing_accuracy:
        print(f"New model accuracy ({new_accuracy:.4f}) is better than existing ({existing_accuracy:.4f}). Replacing model.")
        return True
    elif new_accuracy < existing_accuracy:
        print(f"New model accuracy ({new_accuracy:.4f}) is not better than existing ({existing_accuracy:.4f}). Keeping existing model.")
        print("To force overwrite, use --force-overwrite flag.")
        return False
    
    # If accuracies are equal (e.g., both 1.0), check secondary metrics
    print("Models have identical accuracy. Checking secondary metrics...")
    
    # Check validation loss (lower is better)
    if ('metrics' in new_model_metadata and 'valid_loss' in new_model_metadata['metrics'] and
        'metrics' in existing_metadata and 'valid_loss' in existing_metadata['metrics']):
        new_loss = new_model_metadata['metrics']['valid_loss']
        old_loss = existing_metadata['metrics']['valid_loss']
        
        if new_loss < old_loss:
            print(f"New model has lower validation loss ({new_loss:.5f} vs {old_loss:.5f}). Replacing model.")
            tracker_to_use[model_name]['secondary_improvement'] = f"Lower validation loss: {new_loss:.5f} vs {old_loss:.5f}"
            return True
        elif old_loss < new_loss:
            print(f"Existing model has lower validation loss ({old_loss:.5f} vs {new_loss:.5f}). Keeping existing model.")
            return False
    
    # Check confidence (higher is better)
    if ('metrics' in new_model_metadata and 'confidence' in new_model_metadata['metrics'] and
        'metrics' in existing_metadata and 'confidence' in existing_metadata['metrics']):
        new_conf = new_model_metadata['metrics']['confidence']
        old_conf = existing_metadata['metrics']['confidence']
        
        if new_conf > old_conf:
            print(f"New model has higher confidence ({new_conf:.4f} vs {old_conf:.4f}). Replacing model.")
            tracker_to_use[model_name]['secondary_improvement'] = f"Higher confidence: {new_conf:.4f} vs {old_conf:.4f}"
            return True
        elif old_conf > new_conf:
            print(f"Existing model has higher confidence ({old_conf:.4f} vs {new_conf:.4f}). Keeping existing model.")
            return False
    
    # Check epochs (fewer is better - more efficient training)
    if ('metrics' in new_model_metadata and 'epochs' in new_model_metadata['metrics'] and
        'metrics' in existing_metadata and 'epochs' in existing_metadata['metrics']):
        new_epochs = new_model_metadata['metrics']['epochs']
        old_epochs = existing_metadata['metrics']['epochs']
        
        if new_epochs < old_epochs:
            print(f"New model trained in fewer epochs ({new_epochs} vs {old_epochs}). Replacing model.")
            tracker_to_use[model_name]['secondary_improvement'] = f"Fewer epochs: {new_epochs} vs {old_epochs}"
            return True
        elif old_epochs < new_epochs:
            print(f"Existing model trained in fewer epochs ({old_epochs} vs {new_epochs}). Keeping existing model.")
            return False
    
    # Check model size (smaller is better)
    new_size = new_model_metadata.get('model_size_bytes')
    existing_size = existing_metadata.get('model_size_bytes')
    
    if new_size and existing_size:
        # Only consider size if new model is at least 10% smaller
        if new_size < existing_size * 0.9:
            print(f"New model is significantly smaller ({new_size/1024/1024:.2f} MB vs {existing_size/1024/1024:.2f} MB). Replacing model.")
            tracker_to_use[model_name]['secondary_improvement'] = f"Smaller size: {new_size/1024/1024:.2f}MB vs {existing_size/1024/1024:.2f}MB"
            return True
    
    # If we got here, no clear winner - keep existing model for stability
    print("No significant difference found in secondary metrics. Keeping existing model for stability.")
    return False

def should_rerun_model_by_quality(model_path, quality_threshold):
    """Check if model should be rerun based on quality threshold"""
    if not model_path.exists():
        return True  # Model doesn't exist, so run it
    
    # Load metadata
    from utils.model_metadata_utils import load_model_metadata
    metadata = load_model_metadata(model_path)
    if not metadata or 'metrics' not in metadata:
        return True  # No metrics, so run it
    
    # Get accuracy
    accuracy = metadata['metrics'].get('accuracy')
    if accuracy is None and 'tta_accuracy' in metadata['metrics']:
        accuracy = metadata['metrics'].get('tta_accuracy')
    
    # Get quality category
    quality = get_model_quality_category(accuracy)
    
    # Quality thresholds in descending order
    quality_levels = ["excellent", "good", "acceptable", "mediocre", "poor"]
    
    # If quality is None, rerun it
    if quality is None:
        return True
    
    # Find index of threshold and current quality
    threshold_index = quality_levels.index(quality_threshold)
    quality_index = quality_levels.index(quality)
    
    # If current quality is at threshold or worse, rerun it
    return quality_index >= threshold_index

def format_metrics_with_emoji(metadata):
    """Format metadata with emoji indicators for accuracy"""
    from utils.model_metadata_utils import format_metadata_for_display
    base_display = format_metadata_for_display(metadata)
    
    # If no metadata or metrics, return empty string
    if not metadata or 'metrics' not in metadata:
        return base_display
    
    # Get accuracy from metrics
    accuracy = metadata['metrics'].get('accuracy')
    if accuracy is None and 'tta_accuracy' in metadata['metrics']:
        # Use TTA accuracy if regular accuracy isn't available
        accuracy = metadata['metrics'].get('tta_accuracy')
    
    # Get appropriate emoji based on accuracy
    emoji = get_accuracy_emoji(accuracy)
    
    # Add emoji to the display
    if emoji and base_display:
        return f"{emoji} {base_display}"
    
    return base_display
