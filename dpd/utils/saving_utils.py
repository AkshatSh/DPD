from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
import sys
import h5py
import numpy as np

def save_h5_dict(input_dict: Dict[int, int], h5f: h5py.File, dataset_name: str):
    # map id to val
    dict_np = np.zeros((len(input_dict), 2), dtype=int)
    for i, (inp, val) in enumerate(input_dict.items()):
        dict_np[i] = (inp, val)
    save_h5_np(dict_np, dataset_name)

def load_h5_dict(h5f: h5py.File, dataset_name: str) -> Dict[int, int]:
    dict_np = load_h5_np(h5f, dataset_name)
    res: Dict[int, int] = {}
    for i, entry in enumerate(dict_np):
        inp, out = entry
        res[inp] = out
    return res

def save_h5_np(data: np.ndarray, h5f: h5py.File, dataset_name: str):
    h5f.create_dataset(dataset_name, data=data)

def load_h5_np(h5f: h5py.File, dataset_name: str) -> np.ndarray:
    data = h5f[dataset_name][:]
    return data