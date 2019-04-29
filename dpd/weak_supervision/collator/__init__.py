from typing import (
    List,
    Dict,
)

from .collator import Collator
from .union_collator import UnionCollator
from .intersection_collator import IntersectionCollator

COLLATOR_IMPLEMENTATION: Dict[str, Collator] = {
    'union': UnionCollator,
    'intersection': IntersectionCollator,
}