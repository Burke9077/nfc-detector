"""
Model Performance Utilities

This module provides utilities for tracking and comparing model performance.
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
