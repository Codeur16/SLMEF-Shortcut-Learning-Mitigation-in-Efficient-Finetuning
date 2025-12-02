import yaml
import os

def load_yaml_config(model_name):
    """Loads a YAML configuration file for a given model."""
    config_path = f"configs/models/{model_name}.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config does not exist: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
