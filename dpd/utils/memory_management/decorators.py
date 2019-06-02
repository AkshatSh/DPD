from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Callable,
    Any,
)

import os
import sys
import time
import logging

from dpd.constants import MAX_RETRY, MEMORY_WAIT

from .errors import MemoryRetryError

logger = logging.getLogger(name=__name__)

def memory_retry(function: callable) -> callable:
    def _wrapper(*args, **kwargs) -> Any:
        retry_count: int = 0
        while retry_count < MAX_RETRY:
            try:
                res = function(*args, **kwargs)
                return res
            except MemoryError as e:
                retry_count += 1
                logger.warning(f'encountered memory error, retrying {retry_count} / {MAX_RETRY}')
                time.sleep(MEMORY_WAIT)
        raise MemoryRetryError()
    return _wrapper