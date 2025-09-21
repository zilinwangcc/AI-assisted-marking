# src/loaders/__init__.py
from typing import Literal
import pandas as pd
from .ammore import load as _load_ammore

def load(
    dataset: Literal["ammore"],
    *,
    split: Literal["seed", "full"] = "seed",
    seed_size: int = 50,
    random_state: int = 42
) -> pd.DataFrame:
    
    if dataset == "Math_ShortAns":
        path = "data/SAS-Bench/math_test_student.jsonl"
        return pd.read_json(path, lines=True)

    else:
        raise NotImplementedError(f"Loader for {dataset} not yet implemented")
