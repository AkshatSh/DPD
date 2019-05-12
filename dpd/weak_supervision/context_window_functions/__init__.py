from .window_function import WindowFunction
from .bag_window_function import BagWindowFunction
from .linear_window_function import LinearWindowFunction

WINDOW_FUNCITON_IMPL = {
    'bag': BagWindowFunction,
    'linear': LinearWindowFunction,
}