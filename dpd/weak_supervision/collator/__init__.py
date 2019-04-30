from typing import (
    List,
    Dict,
)

from .collator import Collator
from .union_collator import UnionCollator
from .intersection_collator import IntersectionCollator
from .snorkel_collator import SnorkelCollator
from .metal_collator import SnorkeMeTalCollator

COLLATOR_IMPLEMENTATION: Dict[str, Collator] = {
    'union': UnionCollator,
    'intersection': IntersectionCollator,
    'snorkel': SnorkelCollator,
    'metal': SnorkeMeTalCollator,
}