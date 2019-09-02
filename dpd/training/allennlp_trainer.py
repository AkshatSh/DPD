import logging
import math
import os
import time
import re
import datetime
import traceback
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, NamedTuple

import torch
import torch.optim.lr_scheduler


from allennlp.training.trainer import Trainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class AllenNLPTrainer(Trainer):
    def _restore_checkpoint(self) -> int:
        # no-op for experiments
        return 0