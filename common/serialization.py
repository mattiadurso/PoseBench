"""Serialization helpers shared across PoseBench."""

from typing import Any

import numpy as np


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: A value possibly containing numpy scalars/arrays, nested in dicts/lists.

    Returns:
        The same structure with numpy integers/floats/arrays converted to native
        ``int``/``float``/``list``.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj
