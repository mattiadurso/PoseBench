"""Shared pytest configuration: make the repo root importable.

This mirrors the ``sys.path`` setup the benchmark entry points perform at runtime,
so tests can ``import common``, ``import demo_utils``, ``import matchers.mnn`` etc.
regardless of the working directory pytest is launched from.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
