from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    Callable,
)

import os
import sys
import time
import logging

logger = logging.getLogger(name=__name__)

class MemoryRetryError(MemoryError):
    pass