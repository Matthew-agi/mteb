# HeAR-s1.1 MAEB

This fork adds `matthewagi/HeAR-s1.1` as an MTEB-loadable audio model and includes a cloud runner for `MAEB(beta, audio-only)`.

Model defaults in the fork:

- Hugging Face model: `matthewagi/HeAR-s1.1`
- Revision: `a5776bebff935a81c79720467ae1e10a4effe10e`
- Audio mode: `2.0s` windows, `2.0s` hop, mean pooling
- Embedding head: student head exposed through the published HF model
- Embedding dim: `384`

## Cloud install

From the fork root:

```bash
python -m pip install -U pip
python -m pip install -e '.[audio,timm]'
```

## Smoke test

Run a single task first:

```bash
python scripts/run_hear_s11_maeb.py \
  --device cuda \
  --tasks BeijingOpera \
  --batch-size 32
```

## Full MAEB run

Run the full benchmark with the saved leaderboard settings:

```bash
python scripts/run_hear_s11_maeb.py \
  --device cuda \
  --batch-size 32
```

Important:

- Do not pass model-shape overrides if you want the canonical leaderboard run.
- The canonical run is the model default already baked into the registry entry.
- Non-default overrides such as `--full-clip`, `--no-sliding-window`, `--clip-seconds`, or `--window-hop-seconds` will be tracked as an experiment variant.

## Outputs

The runner writes:

- `run_config.json`
- `model_result.json`
- `overall_results.json`
- results-repo layout under:
  `results/<model>/<revision>/...`

For the default leaderboard run, that layout will be:

```text
<output-dir>/results/matthewagi__HeAR-s1.1/a5776bebff935a81c79720467ae1e10a4effe10e/
```

That directory contains:

- one JSON file per MAEB task
- `model_meta.json`
- `overall_results.json`

## Compatibility fixes in this fork

This fork includes the MAEB-related compatibility fixes needed to run cleanly with the published `HeAR-s1.1` wrapper:

- `dataset_transform(num_proc=...)` compatibility for task classes that do not accept `num_proc`
- `AudioDecoder`-safe validation for `VoxPopuliLanguageID`
- `AudioDecoder`-safe validation for `VoxPopuliAccentID`

## PR flow

Upstream `mteb` PR:

- include the model implementation in `mteb/models/model_implementations/hear_s11_models.py`
- include the generic/task compatibility fixes in `mteb/_audio.py`, `mteb/abstasks/abstask.py`, and the two VoxPopuli task files
- include the tests under `tests/test_models/` and `tests/test_tasks/`

Fork-only convenience files:

- `scripts/run_hear_s11_maeb.py`
- `README_HEAR_S11_MAEB.md`

Recommended sequence:

1. Push the upstream-facing model implementation, compatibility fixes, and tests from this fork in your `mteb` PR.
2. Keep `scripts/run_hear_s11_maeb.py` locally in your fork for the cloud benchmark run.
3. Run the benchmark on the cloud machine with `scripts/run_hear_s11_maeb.py`.
4. Use the generated `results/...` subtree for the `embeddings-benchmark/results` PR.
