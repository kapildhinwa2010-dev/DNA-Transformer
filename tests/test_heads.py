import numpy as np
import pytest

from dna_transformer import encode_heads, decode_heads, heads_max_abs_error


def _matrix(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((8, 16))


def test_heads_roundtrip():
    W = _matrix()
    records = encode_heads(W, num_heads=4, bits=8)
    assert len(records) == 4
    out = decode_heads(records)
    assert out.shape == W.shape
    assert np.abs(W - out).max() < 0.05


def test_heads_lower_error_than_global():
    # Heads with very different ranges: per-head scales should beat one global scale
    rng = np.random.default_rng(1)
    blocks = [rng.standard_normal((2, 8)) * s for s in (0.01, 1.0, 50.0, 1000.0)]
    W = np.vstack(blocks)
    per_head = heads_max_abs_error(W, num_heads=4, bits=8)
    assert per_head < 1000.0 * 0.05


def test_heads_rejects_bad_num_heads():
    with pytest.raises(ValueError):
        encode_heads(_matrix(), num_heads=3)


def test_heads_rejects_1d():
    with pytest.raises(ValueError):
        encode_heads(np.ones(8), num_heads=2)


def test_decode_heads_rejects_empty():
    with pytest.raises(ValueError):
        decode_heads([])
