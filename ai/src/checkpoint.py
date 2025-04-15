import os
import pickle
import jax
import numpy as np
from .model import QuantizedWeight8bit

def fast_unpickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def fast_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def restore(checkpoint_path: str, state_shapes):
    ckpt_path = os.path.join(checkpoint_path, "ckpt-0")
    print(f"Loading checkpoint at {ckpt_path}")
    return fast_unpickle(ckpt_path)

def save(state, checkpoint_path: str):
    ckpt_path = os.path.join(checkpoint_path, "ckpt-0")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    fast_pickle(state, ckpt_path)