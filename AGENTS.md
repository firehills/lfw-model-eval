# Project guidelines

## Architecture and scope

- This is a standalone Python app suite, not an installable package. Keep the
  three root CLIs thin and shared logic under `face_eval/`.
- Use Python 3.12 in the CPU devcontainer (`.devcontainer/`) or the GPU one
  (`.devcontainer/gpu/`, onnxruntime-gpu on CUDA 12/cuDNN 9). Run development and
  validation there; do not install the ML stack into the WSL host Python.
- Read [README.md](README.md) for the protocol, binary format, configuration
  schema and metric definitions before changing behavior.
- ONNX/TFLite inference on CPU or GPU (`--device`). ONNX GPU uses
  `CUDAExecutionProvider` and must verify it is the active provider; TFLite GPU
  requires an explicit delegate library. GPU requests must never fall back to CPU.
- RetinaFace returns numpy scalars; convert to Python types before writing JSON.

## Pipeline contracts

- Preserve `Preprocessor.process(path)` as the shared model-ready tensor entry
  point. Inject detectors using the `Detector` protocol; keep RetinaFace imports
  lazy so offline tests do not download detector weights.
- Put model-specific layout, input size, channel order, normalization,
  quantization and target landmarks in `ModelConfig`/JSON profiles, not
  model-name conditionals. Example profiles must not imply universal export
  compatibility.
- Landmark order is image-left eye, image-right eye, nose, image-left mouth,
  image-right mouth. Custom coordinates are in output pixels.
- Binary tensors are little-endian, contiguous C-order, batch 1, three channels.
  Keep manifest shape, dtype, hashes and serialized bytes consistent.
- Selection is lexicographic: first X enrolment, next Y test. Never shuffle,
  reuse a source across splits or silently exclude processing failures.
- Preserve explicit threshold semantics: similarities accept >=, distances
  accept <=. Never tune a threshold on the evaluation set.
- Verification compares every probe to every identity template. Keep rank-1
  identification separate; do not label these results official LFW pair scores
  or open-set rejection performance.
- Time only synchronous embedding calls, with warmup and other pipeline work
  excluded as documented. Keep timing metadata accurate if adapters change.

## Reliability and data handling

- Validate shapes, dtypes, finite/nonzero embeddings and input/output
  quantization; reject unsupported model contracts rather than guessing.
- No silent CPU fallbacks, skipped faces, fabricated defaults or swallowed
  failures. Invalid input must produce an explanatory error/nonzero exit.
- Never overwrite completed datasets, prepared tensors or reports implicitly.
  Keep archive path/link/special-file/size checks and staged dataset publishing.
- Do not commit biometric images, binaries, embeddings, reports or model
  weights. Use the git-ignored `data/`, `models/`, `artifacts/`, `reports/`.
- Do not automatically download recognition weights or change dataset sources
  without documenting provenance, checksums and licensing considerations.

## Validation

Run inside the devcontainer:

```bash
python -m pytest tests -q
python -m ruff check .
python -m compileall -q face_eval download_dataset.py prepare_tensors.py evaluate.py
python -m pip check
```

- Use the smallest relevant test files during iteration; run the full suite for
  shared pipeline/runtime changes.
- Keep tests offline using generated images/models and injected detections.
  For pipeline changes also run a real smoke test on a few LFW identities
  (`prepare_tensors.py --max-identities`) with a real model, since mocked
  detectors cannot catch every RetinaFace integration issue. GPU changes must be
  checked in the GPU image with `docker run --gpus all`.
  Preserve real ONNX and float/integer TFLite integration coverage.
- Add exact assertions for threshold boundaries, score direction, confusion
  counts, split leakage and tensor byte shape/dtype when changing those areas.
- Update [README.md](README.md), example profiles and related tests for any
  CLI/schema/protocol change. Change pinned requirements only when necessary
  and verify the Docker build and `pip check` afterwards.
