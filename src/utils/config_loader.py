"""
Configuration loading utilities
"""

import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations (override takes precedence)"""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged

def load_experiment_config(model_name: str, task_name: str, exp_type: str, config_dir: str = "configs") -> Dict[str, Any]:
    """Load experiment configuration"""
    # Load model config
    model_config_path = os.path.join(config_dir, "models", f"{model_name}.yaml")
    model_config = load_config(model_config_path)
    
    # Load task config
    task_config_path = os.path.join(config_dir, "tasks", f"{task_name}.yaml")
    task_config = load_config(task_config_path)
    
    # Load experiment config if exists
    exp_config_path = os.path.join(config_dir, "experiments", exp_type, f"{model_name}_{task_name}.yaml")
    exp_config = {}
    if os.path.exists(exp_config_path):
        exp_config = load_config(exp_config_path)
    
    # Merge configurations
    config = merge_configs(model_config, task_config)
    config = merge_configs(config, exp_config)
    
    # Add metadata
    config["experiment"] = {
        "model": model_name,
        "task": task_name,
        "type": exp_type
    }
    
    return config