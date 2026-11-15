# DNA-Transformer

Encode/decode codec for weight matrices (genes).

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

![CI](https://github.com/kapildhinwa2010-dev/DNA-Transformer/actions/workflows/ci.yml/badge.svg)
![CodeQL](https://github.com/kapildhinwa2010-dev/DNA-Transformer/actions/workflows/codeql.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- `encode()` / `decode()` quantized codec (1-16 bits)
- `GeneRecord` serializes to plain dicts
- `codec_stats()` reports ratio and max error

## 📦 Install

```bash
pip install -r requirements.txt
```

## 🚀 Quick start

```python
from dna_transformer import encode, decode, codec_stats

record = encode(W, bits=8)
print(codec_stats(W, bits=8)['ratio'])
```

## 🧪 Tests

```bash
pytest tests/ -v
```

## 📄 License

MIT — see [LICENSE](LICENSE).
