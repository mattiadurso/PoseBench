"""Tests for common.serialization.convert_numpy_types."""

import numpy as np

from common.serialization import convert_numpy_types


def test_scalar_conversions():
    """convert_numpy_types turns numpy scalars into native int/float."""
    assert convert_numpy_types(np.int64(3)) == 3
    assert isinstance(convert_numpy_types(np.int64(3)), int)
    assert convert_numpy_types(np.float32(1.5)) == 1.5
    assert isinstance(convert_numpy_types(np.float32(1.5)), float)


def test_ndarray_conversion():
    """convert_numpy_types converts an ndarray into a nested list."""
    # Regression: a previous copy used isinstance(obj, np.array), which raises
    # TypeError because np.array is a function, not a type. The canonical version
    # uses np.ndarray and must convert arrays to lists.
    arr = np.array([[1, 2], [3, 4]])
    out = convert_numpy_types(arr)
    assert out == [[1, 2], [3, 4]]
    assert isinstance(out, list)


def test_nested_structures():
    """convert_numpy_types recurses through nested dicts and lists."""
    obj = {
        "a": np.int64(1),
        "b": [np.float64(2.0), np.array([3, 4])],
        "c": {"d": np.float32(5.5)},
        "e": "str",
    }
    out = convert_numpy_types(obj)
    assert out == {"a": 1, "b": [2.0, [3, 4]], "c": {"d": 5.5}, "e": "str"}


def test_passthrough_native_types():
    """convert_numpy_types passes native int/str/None through unchanged."""
    assert convert_numpy_types(7) == 7
    assert convert_numpy_types("x") == "x"
    assert convert_numpy_types(None) is None
