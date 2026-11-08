"""S2c - one place to ask for a model by name (used by notebooks and run scripts).

Models that belong to an approach live in that approach's folder and are imported lazily."""
from stages.s2_models.cnn import CNN
from stages.s2_models.dnn import DNN


def get_model_factory(name):
    """Return a zero-argument callable that builds a fresh model."""
    if name == "dnn":
        return DNN
    if name == "cnn":
        return CNN
    if name == "moe":
        from moe.model import MoE
        return MoE
    if name == "dnn_2023":
        from baseline_2023.model import DNN2023
        return DNN2023
    raise ValueError(f"unknown model {name!r}; choose dnn | cnn | moe | dnn_2023")
