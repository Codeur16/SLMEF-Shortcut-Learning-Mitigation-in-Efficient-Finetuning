from .flant5 import FlanT5Model

def load_model(name: str, exp_type: str):
    registry = {
        "flant5": FlanT5Model
    }

    if name not in registry:
        raise ValueError(f"Modèle inconnu : {name}")

    # BaseModel signature expects (model_name, exp_type, checkpoint=None)
    return registry[name](name, exp_type)
