from .cwr_linear import CWRLinear
from .cwr_knn import CWRkNN

CONTEXTUAL_FUNCTIONS_IMPL = {
    'cwr_linear': CWRLinear,
    'cwr_knn': CWRkNN,
}