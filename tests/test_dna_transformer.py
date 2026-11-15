import numpy as np
import pytest

from dna_transformer import GeneRecord, codec_stats, decode, encode


def _matrix(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((8, 16))


def test_roundtrip_low_error():
    W = _matrix()
    record = encode(W, bits=8)
    recovered = decode(record)
    assert np.abs(W - recovered).max() < 0.05


def test_roundtrip_exact_for_integers_16bit():
    W = np.arange(12, dtype=np.float64).reshape(3, 4)
    record = encode(W, bits=16)
    assert np.abs(W - decode(record)).max() < 0.001


def test_encode_rejects_1d():
    with pytest.raises(ValueError):
        encode(np.ones(5))


def test_encode_rejects_empty():
    with pytest.raises(ValueError):
        encode(np.array([]))


def test_encode_rejects_bad_bits():
    with pytest.raises(ValueError):
        encode(_matrix(), bits=0)
    with pytest.raises(ValueError):
        encode(_matrix(), bits=17)


def test_codes_dtype_u8():
    record = encode(_matrix(), bits=8)
    assert record.codes.dtype == np.uint8


def test_codes_dtype_u16():
    record = encode(_matrix(), bits=12)
    assert record.codes.dtype == np.uint16


def test_constant_matrix_roundtrip():
    W = np.full((4, 4), 7.25)
    record = encode(W, bits=8)
    assert np.allclose(decode(record), W)


def test_to_dict_from_dict_roundtrip():
    W = _matrix()
    record = encode(W, bits=8)
    clone = GeneRecord.from_dict(record.to_dict())
    assert np.array_equal(decode(clone), decode(record))


def test_decode_rejects_non_record():
    with pytest.raises(TypeError):
        decode({"codes": 1})


def test_codec_stats_ratio_above_one():
    W = _matrix()
    stats = codec_stats(W, bits=8)
    assert stats["ratio"] > 1.0
    assert stats["max_abs_error"] < 0.05
    assert stats["encoded_bytes"] < stats["original_bytes"]


def test_gene_record_nbytes():
    record = encode(_matrix(), bits=8)
    assert record.nbytes() == record.codes.nbytes + 16
