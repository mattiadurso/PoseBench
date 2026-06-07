"""Headless smoke tests for the demo plotting helpers.

These cannot check visual output, but they run the drawing code end-to-end on the
non-interactive Agg backend to catch crashes across the refactor.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import demo_utils  # noqa: E402


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    # Avoid opening windows / blocking on plt.show() during tests.
    monkeypatch.setattr(demo_utils.plt, "show", lambda *a, **k: None)


def test_plot_imgs_runs():
    imgs = [np.random.rand(16, 16, 3) for _ in range(3)]
    demo_utils.plot_imgs(imgs, titles=["a", "b", "c"], rows=1)


def test_plot_imgs_and_kpts_runs():
    img1 = np.random.rand(20, 24, 3).astype(np.float32)
    img2 = np.random.rand(18, 24, 3).astype(np.float32)
    kpt1 = np.random.rand(10, 2).astype(np.float32) * 15
    kpt2 = np.random.rand(10, 2).astype(np.float32) * 15
    demo_utils.plot_imgs_and_kpts(
        img1, img2, kpt1, kpt2, matches=True, scatter=True, sample_points=5
    )
