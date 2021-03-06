from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Any,
    Union,
)

import os
import sys
import h5py
import numpy as np
import pickle

from .saving_utils import (
    load_h5_dict,
    load_h5_np,
    save_h5_dict,
    save_h5_np,
)

class SaveFile(object):
    def __init__(
        self,
        file_name: str
    ):
        pass
    
    def load_dict(
        self,
        key: str,
    ) -> Dict[int, int]:
        raise NotImplementedError()
    
    def load_np(
        self,
        key: str,
    ) -> np.ndarray:
        raise NotImplementedError()
    
    def save_dict(
        self,
        item: Dict[int, int],
        key: str,
    ):
        raise NotImplementedError()
    
    def save_np(
        self,
        item: np.ndarray,
        key: str,
    ):
        raise NotImplementedError()
    
    def close(self):
        raise NotImplementedError()

class H5SaveFile(SaveFile):
    def __init__(
        self,
        file_name: str,
    ):
        self.file_name = file_name
        self.file = h5py.File(file_name)
    
    def load_dict(
        self,
        key: str,
    ) -> Dict[int, int]:
        return load_h5_dict(self.file, key)
    
    def load_np(
        self,
        key: str,
    ) -> np.ndarray:
        return load_h5_np(self.file, key)
    
    def save_dict(
        self,
        item: Dict[int, int],
        key: str,
    ):
        save_h5_dict(input_dict=item, h5f=self.file, dataset_name=key)
    
    def save_np(
        self,
        item: np.ndarray,
        key: str,
    ):
        save_h5_np(data=item, h5f=self.file, dataset_name=key)

    def close(self):
        self.file.close()

class PickleSaveFile(SaveFile):
    def __init__(
        self,
        file_name: str,
    ):
        self.file_name = file_name
        self.data: Dict[str, Union[Dict[Any, Any], np.ndarray]] = {}
        if os.path.exists(self.file_name):
            self.file = open(self.file_name, 'rb')
            self.data = pickle.load(self.file)
        else:
            self.file = open(self.file_name, 'wb')
    
    def load_dict(
        self,
        key: str,
    ) -> Dict[Any, Any]:
        return self.data[key]
    
    def load_np(
        self,
        key: str,
    ) -> np.ndarray:
        return self.data[key]
    
    def save_dict(
        self,
        item: Dict[Any, Any],
        key: str,
    ):
        self.data[key] = item
    
    def save_np(
        self,
        item: np.ndarray,
        key: str,
    ):
        self.data[key] = item

    def close(self):
        pickle.dump(self.data, self.file)
        self.file.close()