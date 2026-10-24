# Third-party code

This repository includes code adapted from two open-source projects. Their licenses are in
[`LICENSES/`](LICENSES/).

## LLM_eval: Tang et al. (2025)

- **Source:** <https://github.com/amberT15/LLM_eval>, `chip-clip-seq/representation_perf.py`
- **License:** MIT, Copyright (c) 2024 amberT15. See [`LICENSES/MIT-LLM_eval.txt`](LICENSES/MIT-LLM_eval.txt).
- **Used in:**
  - `src/ctcf/models.py`: the `chip_cnn` baseline network. The unused BatchNorm + 1×1 conv
    branch is removed, and the network is ported to the `tf_keras` package.
  - `tests/test_models.py`: a verbatim copy of `chip_cnn`, used to check that the two networks
    are identical.

## HyenaDNA: Nguyen et al. (2023)

- **Source:** <https://github.com/HazyResearch/hyena-dna>, `standalone_hyenadna.py`. The
  LLM_eval repository uses an identical copy.
- **License:** Apache License 2.0. See [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).
- **Used in:** `src/ctcf/hyenadna_standalone.py`, the HyenaDNA model and character tokenizer.
  The code is unchanged apart from a header naming its source and license.
