# DNA-Transformer

[![CI](https://github.com/kapildhinwa2010-dev/DNA-Transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-Transformer/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/DNA-Transformer/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-Transformer/actions/workflows/codeql.yml)


DNA quantized codec tuned for transformer weight matrices: per-head encoding.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

DNA-Transformer is the transformer-specialized member of the DNA codec
family (lineage: the private DNA repository). encode() quantizes a
float matrix into a GeneRecord (codes + scale metadata); decode()
reconstructs it. The transformer-specific part: encode_heads() splits
an attention weight matrix into equal head blocks and encodes each
with its OWN scale, which lowers reconstruction error versus one
global scale when heads have different value ranges.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
from dna_transformer import encode, decode, codec_stats, encode_heads, decode_heads

record = encode(W, bits=8)
print(codec_stats(W, bits=8)["ratio"])
records = encode_heads(W_attn, num_heads=8, bits=8)
W2 = decode_heads(records)
```

## API

- encode(matrix, bits) / decode(record) / codec_stats(matrix, bits)
- encode_heads(matrix, num_heads, bits) / decode_heads(records)
- heads_max_abs_error(matrix, num_heads, bits)

## Honest results and limitations

- Lossy: error equals the quantization step; codec_stats() reports
  max_abs_error so you can check it.
- "Ratio" is float32-bytes / encoded-bytes ARITHMETIC - it counts
  stored codes, not a real serialized file with headers.
- encode_heads() requires num_heads to divide the row count; all
  heads are equal-sized splits, not named attention heads.
- No per-layer fitting: the reserv.dean-part-2 log records that
  generated-weight reconstructions of trained models were not usable
  without per-layer fitting.

## License

MIT - see [LICENSE](LICENSE).
