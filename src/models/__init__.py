# from .flant5 import FlanT5Model

# def load_model(name: str, exp_type: str):
#     registry = {
#         "flant5": FlanT5Model
#     }

#     if name not in registry:
#         raise ValueError(f"Modèle inconnu : {name}")

#     # BaseModel signature expects (model_name, exp_type, checkpoint=None)
#     return registry[name](name, exp_type)


from .factory import ModelFactory
from .bert_model import BERTModel
from .roberta_model import RoBERTaModel
from .flant5_model import FlanT5Model
from .llama_model import LLaMAModel
from .mistral_model import MistralModel

__all__ = [
    "ModelFactory",
    "BERTModel",
    "RoBERTaModel", 
    "FlanT5Model",
    "LLaMAModel",
    "MistralModel",
]