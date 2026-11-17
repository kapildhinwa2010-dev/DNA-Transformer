# DNA-Transformer

DNA-Transformer is a small NumPy-based library for lossy quantization of two-dimensional numeric matrices. It stores quantized codes with range and shape metadata and can encode equal-sized, contiguous row blocks independently. Despite the repository name, it does not implement a Transformer model or integrate with a training or inference framework.

## Project status

The repository contains a single Python module, tests, and packaging metadata (version `0.1.0`). Treat it as an experimental matrix-quantization utility, not as a validated model-compression system: the project does not include a Transformer implementation, trained-model evaluation, or a benchmark suite.

## Implemented functionality

- `encode(matrix, bits=8)` converts a non-empty 2-D array to a `GeneRecord`. The implementation uses the matrix minimum and maximum to linearly map values to integer codes; `bits` must be from 1 through 16.
- `decode(record)` reconstructs an array with the recorded shape from those codes and range metadata. This is lossy quantization in general; it is not a lossless round trip.
- `GeneRecord` carries the codes, lower bound, scale, shape, and bit width. `to_dict()` and `from_dict()` provide in-memory conversion helpers; the project does not define a file serialization format.
- `encode_heads(matrix, num_heads, bits=8)` divides the matrix into equal contiguous row blocks and encodes each block with its own range. The row count must be divisible by `num_heads`. `decode_heads(records)` stacks decoded blocks vertically.
- `codec_stats(matrix, bits=8)` reports code-based size estimates, a ratio, and maximum absolute reconstruction error. `heads_max_abs_error(...)` reports the maximum absolute error for the row-block round trip.

## Requirements

- Python 3.9 or newer, as declared in `pyproject.toml`.
- NumPy 1.24 or newer.

## Installation

From the repository root, install the project in editable mode:

```bash
python -m pip install -e .
```

The module can then be imported as `dna_transformer`. There is no command-line interface.

## Example

```python
import numpy as np
from dna_transformer import decode, encode, encode_heads, decode_heads, codec_stats

weights = np.array([[0.0, 0.5], [1.0, -0.5]], dtype=np.float32)
record = encode(weights, bits=8)
reconstructed = decode(record)
print(reconstructed)
print(codec_stats(weights, bits=8))

# Example of equal-row-block encoding; the row count must divide evenly.
blocks = encode_heads(weights, num_heads=2, bits=8)
reconstructed_blocks = decode_heads(blocks)
```

## Tests and CI

At commit `453445f` (the commit reviewed for this draft), the local test suite was run with Python 3.12.3: `python -m pytest -q` completed with **17 passed**. The module and tests also compiled with `python -m compileall -q dna_transformer.py tests`, and the import/encode/decode shape smoke check configured in CI passed locally. These are local results; the GitHub Actions run status was not checked. The CI workflow is configured for Python 3.10, 3.11, and 3.12, but this review did not run the suite on all three versions.

To install the test dependencies and run the suite yourself:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Limitations

- Quantization is lossy and uses one linear range per encoded matrix or row block. The code does not implement per-layer calibration, model-aware quantization, or a trained-model workflow.
- `encode_heads` uses equal contiguous row splits; it does not discover or verify actual attention-head boundaries. Its row-block API can also be used on other matrices, but the source provides no Transformer-specific model integration.
- `codec_stats` is an arithmetic estimate, not a measured on-disk compression ratio: it assumes four bytes per input element and estimates encoded storage as the code-array bytes plus 16 bytes of metadata. No actual serialized representation or file-size measurement is provided.
- The tests cover focused unit cases and a synthetic matrix with differing block ranges. The repository reports no real-model accuracy evaluation or performance benchmarks.

## License

The repository's `LICENSE` file contains the MIT License (copyright 2026 Kapildhinwa2010-dev). See [LICENSE](LICENSE) for the full terms.
