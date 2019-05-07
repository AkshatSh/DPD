from .window_function import WindowFunction
from .bag_window_function import BagWindowFunction
from .linear_window_function import LinearWindowFunction
from .utils import get_context_range, get_context_window

WINDOW_FUNCITON_IMPL = {
    'bag': BagWindowFunction,
    'linear': LinearWindowFunction,
}