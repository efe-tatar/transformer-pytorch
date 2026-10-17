"""Spec for step 7 (yours): warmup + cosine learning-rate schedule."""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.schedule import get_lr

KW = dict(warmup_steps=10, max_steps=110, max_lr=1e-3, min_lr=1e-4)


def test_warmup_is_linear_and_never_zero():
    assert get_lr(0, **KW) == pytest.approx(1e-4)
    assert get_lr(4, **KW) == pytest.approx(5e-4)
    assert get_lr(9, **KW) == pytest.approx(1e-3)


def test_peak_at_end_of_warmup():
    assert get_lr(10, **KW) == pytest.approx(1e-3)


def test_cosine_midpoint_is_mean_of_max_and_min():
    assert get_lr(60, **KW) == pytest.approx(5.5e-4)


def test_reaches_and_stays_at_min_lr():
    assert get_lr(110, **KW) == pytest.approx(1e-4)
    assert get_lr(500, **KW) == pytest.approx(1e-4)


def test_monotonic_decay_after_warmup():
    lrs = [get_lr(s, **KW) for s in range(10, 120)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
