from typing import (
    Any,
)

import torch

def no_grad(function: callable) -> callable:
    def _wrapper(*args, **kwargs) -> Any:
        with torch.no_grad():
            return function(*args, **kwargs)
    return _wrapper