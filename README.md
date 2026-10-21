# Face recognition model evaluation

A Docker-isolated (CPU or GPU) Python suite for evaluating **ONNX** and **TensorFlow
Lite** face embedding models, including ArcFace and MobileFaceNet exports. Run the
apps directly with `python`; there is no package installation step for this repo.

The pipeline downloads a dataset, selects disjoint enrolment/test images,
detects and aligns faces with RetinaFace, saves model-ready binary tensors, then
evaluates embeddings against enrolment identity templates.

## Scope and interpretation

- Initially supports LFW (original and funneled) and other identity-folder datasets.
- JPEG (`.jpg`, `.jpeg`) and BMP, including uppercase extensions.
- Batch 1, three channels, square inputs of any positive size, e.g. 112, 240 or 320.
- NCHW/NHWC, RGB/BGR, per-channel normalization, float32/int8/uint8 tensors.
- Arbitrary vector embedding dimensions, including 128, 192 and 512.
- Cosine similarity, dot product, Euclidean distance and Manhattan distance.
- CPU or GPU inference (`--device cpu|gpu`). ONNX GPU runs use the CUDA execution
  provider; TFLite GPU runs use a user-supplied GPU delegate library. A GPU run
  never falls back to CPU silently; it fails with an explanation instead.
- Uses enrolment-template verification and closed-set rank-1 identification.
  **This is not the official LFW pairs/10-fold protocol**, and its results must
  not be compared directly to published LFW benchmark accuracy.
- A recognition model and its correct preprocessing contract must be supplied by
  the user. Example configurations are starting points, not guarantees for all
  exports carrying an ArcFace/MobileFaceNet name.

## 1. Development environment (WSL2 + devcontainer)

Prerequisites on the host:

1. WSL2 with Docker Engine, or Docker Desktop with WSL integration enabled.
2. VS Code with the WSL and Dev Containers extensions.
3. Clone/open this repository **inside the WSL Linux filesystem** for better I/O.

Open the folder in VS Code from WSL and run **Dev Containers: Reopen in
Container**. The image installs the pinned direct dependencies from
[`requirements-dev.txt`](requirements-dev.txt), including TensorFlow, TensorFlow
Lite, ONNX Runtime, RetinaFace and the test tools. The non-root `vscode` user owns
normal development processes. No GPU, privileged mode or Docker socket mount is
needed. Model weights and datasets are not baked into the image.

The TensorFlow dependency is large; allow several GB for Docker layers and
runtime data. The image intentionally hides GPUs. Only official Debian apt sources
are refreshed, avoiding unrelated inherited repository key failures.

### GPU devcontainer (optional)

[`.devcontainer/gpu/`](.devcontainer/gpu/devcontainer.json) is a second
configuration based on `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` with
`onnxruntime-gpu` ([`requirements-gpu.txt`](requirements-gpu.txt)) replacing the
CPU ONNX Runtime wheel. It needs a recent NVIDIA Windows driver with WSL2 GPU
support and Docker GPU access (`docker run --rm --gpus all ubuntu nvidia-smi`
must list your GPU). In VS Code choose **Dev Containers: Reopen in Container** and
pick *Face model evaluation (GPU)*. From a terminal:

```bash
docker build -f .devcontainer/gpu/Dockerfile -t lfw-model-eval:gpu .
docker run --rm -it --gpus all \
  --user "$(id -u):$(id -g)" -e HOME=/tmp/face-eval-home \
  -v "$PWD:/workspaces/lfw-model-eval" -w /workspaces/lfw-model-eval \
  lfw-model-eval:gpu bash
```

RetinaFace preprocessing still runs on CPU in both images (TensorFlow is
hidden from the GPU); only recognition-model inference is accelerated. All
commands below work identically in either image.

### CPU image from a terminal

Alternatively, build/run the CPU image from a WSL terminal:

```bash
docker build -f .devcontainer/Dockerfile -t lfw-model-eval:cpu .
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/face-eval-home \
  -v "$PWD:/workspaces/lfw-model-eval" \
  -w /workspaces/lfw-model-eval \
  lfw-model-eval:cpu bash
```

Run all following `python` commands **inside the container**. A host Python
environment is unnecessary. The project folder is bind-mounted, so outputs
persist after the container exits.

RetinaFace downloads its detector weights on first use into
`$HOME/.deepface/weights` (network access required). In the VS Code container this
cache lives in the container user's home; a rebuild may require downloading
again. For repeatable/offline runs preserve that cache using a Docker volume,
record the detector weight checksum, and supply it before running preprocessing.
The detector library manages that download; recognition weights are never
automatically downloaded.

## 2. Download a dataset

```bash
python download_dataset.py --dataset lfw --output data/lfw-original
```

This downloads the original LFW archive (roughly 200 MB) from the Figshare mirror
used by scikit-learn, verifies its known SHA-256, then extracts it:

```text
data/lfw-original/
  download.json
  lfw/
    Aaron_Eckhart/
      Aaron_Eckhart_0001.jpg
    ...
```

Use `data/lfw-original/lfw` as the dataset root, **not** the extraction directory.
For the funneled variant:

```bash
python download_dataset.py --dataset lfw-funneled --output data/lfw-funneled
# Dataset root: data/lfw-funneled/lfw_funneled
```

Other datasets can be local identity-folder trees, or HTTPS tar/zip archives:

```bash
python download_dataset.py \
  --url https://your-host.example/your-dataset.zip \
  --sha256 YOUR_64_CHARACTER_SHA256 \
  --output data/custom
```

The custom URL overrides the built-in URL. Omitting a custom checksum emits a
warning; the observed checksum is still recorded in `download.json`, but cannot
establish authenticity. The archive folder structure is preserved. Point the
preparation app at the directory whose **immediate children are identities**:

```text
dataset-root/
  identity-one/
    0001.jpg
    0002.bmp
  identity-two/
    0001.jpg
    0002.jpg
```

Extraction rejects path traversal, links, special files and duplicate paths.
Downloads/extracted declared file sizes are each limited to 4 GiB by default
(`--max-bytes` overrides both). Extraction is staged and published only after
success. Existing destinations are never overwritten. Only download archives
from trusted sources; size limits do not replace archive parser isolation.

## 3. Configure the model's preprocessing

Start from [`configs/arcface_112_nchw.json`](configs/arcface_112_nchw.json) or
[`configs/mobilefacenet_112_nhwc.json`](configs/mobilefacenet_112_nhwc.json).
**Check the model's training/export documentation**, especially alignment,
normalization and channel order. Input metadata alone cannot reveal those.

| Field | Meaning |
|---|---|
| `name` | Human-readable preprocessing profile identifier |
| `size` | Positive integer height = width |
| `layout` | `NCHW` = `[1,3,H,W]`; `NHWC` = `[1,H,W,3]` |
| `color` | `RGB` or `BGR` |
| `mean`, `std` | Three channel values, in the **selected color order** |
| `dtype` | `float32` (default), `uint8` or `int8` |
| `quantization_scale`, `quantization_zero_point` | Required for integer tensors; forbidden for float32 |
| `alignment` | `arcface` or `custom` |
| `landmarks` | For `custom` only: five `[x,y]` targets in **output pixel coordinates** |
| `detection_threshold` | RetinaFace confidence cutoff in `(0,1]`, default `0.9` |
| `face_selection` | `largest` (default) or `single` (reject multiple detections) |

Processing order:

1. Decode the image, apply EXIF orientation and convert to three-channel RGB.
2. Pass BGR pixels to the CPU RetinaFace detector for bounding boxes and five
   landmarks. Select the largest bounding-box area, or require exactly one face.
3. Compute a least-squares, rotation/scale/translation similarity transform to
   the configured landmarks. Warp to the target size using bilinear interpolation
   and black borders. This is planar alignment, **not** 3D pose correction.
4. Apply RGB/BGR ordering.
5. Compute `real = (pixel - mean) / std`, broadcasting over channels.
6. For integer input, compute
   `q = clip(round(real / scale) + zero_point, dtype_min, dtype_max)`.
   Rounding is nearest-even; clipping intentionally saturates out-of-range values.
7. Arrange layout, add batch dimension and make a contiguous tensor.

The ArcFace template at 112 pixels is:

```text
image-left eye   [38.2946, 51.6963]
image-right eye  [73.5318, 51.5014]
nose             [56.0252, 71.7366]
image-left mouth [41.5493, 92.3655]
image-right mouth[70.7299, 92.2041]
```

`alignment=arcface` scales those coordinates by `size / 112`. That is an explicit
choice, not a claim that every 240/320 model uses the same alignment. For models
with another template, set `alignment=custom` and supply `landmarks` in exactly
the order above. RetinaFace eye/mouth pairs are ordered by image x-coordinate;
this assumes approximately upright faces. No face, invalid landmarks or multiple
faces under `single` are errors; there is no silent crop/resize fallback.

Example integer input fields (use the **actual exported model** parameters):

```json
{
  "name": "example-int8",
  "size": 112,
  "layout": "NHWC",
  "color": "RGB",
  "mean": [127.5, 127.5, 127.5],
  "std": [127.5, 127.5, 127.5],
  "dtype": "int8",
  "quantization_scale": 0.0078125,
  "quantization_zero_point": 0,
  "alignment": "arcface"
}
```

### Reuse in other applications

[`Preprocessor.process`](face_eval/preprocessing.py) is the common entry point:

```python
from face_eval.config import ModelConfig
from face_eval.preprocessing import Preprocessor, RetinaFaceDetector

config = ModelConfig.load("configs/arcface_112_nchw.json")
detector = RetinaFaceDetector(config.detection_threshold)
preprocessor = Preprocessor(config, detector)
tensor, face = preprocessor.process("path/to/image.bmp")
# tensor is a contiguous numpy array, ready for the configured model.
# face contains bbox, five image-coordinate keypoints and detector confidence.
```

Keep this repo on `PYTHONPATH` when reusing these modules elsewhere. Inject any
detector implementing `detect(bgr_image) -> list[Face]` to replace RetinaFace
without changing alignment/tensor logic. Landmark order must match the template.

## 4. Select images and save tensors

```bash
python prepare_tensors.py \
  --dataset-root data/lfw-original/lfw \
  --config configs/arcface_112_nchw.json \
  --enrolment 2 --testing 3 \
  --output artifacts/arcface
```

For each identity, supported filenames are sorted **lexicographically,
case-sensitively**. The first X files are enrolled, the next Y files are probes;
the rest are unused. LFW's zero-padded names naturally sort in numeric order.
For custom datasets, zero-pad numeric filenames if numeric order matters.
Identity directories are sorted too; nested directories and symlinks are ignored.
There is no shuffle or random split.

Without `--identities`, identities with fewer than X+Y images are excluded and
listed with counts in the manifest. To explicitly select identities:

```bash
python prepare_tensors.py \
  --dataset-root data/lfw-original/lfw \
  --config configs/arcface_112_nchw.json \
  --identities George_W_Bush Tony_Blair \
  --enrolment 2 --testing 3 \
  --output artifacts/arcface-two-identities
```

An explicitly requested identity with insufficient images is an error.
To take a quick subset without listing names, use `--max-identities N` to keep
the first N eligible identities in alphabetical order. Per-image progress is
printed unless `--quiet` is given.
Evaluation needs **at least two identities**, each present in both splits.

Output structure:

```text
artifacts/arcface/
  manifest.json
  enrolment/George_W_Bush/George_W_Bush_0001.jpg.bin
  enrolment/George_W_Bush/George_W_Bush_0002.jpg.bin
  test/George_W_Bush/George_W_Bush_0003.jpg.bin
  ...
```

Binary files are headerless, little-endian, C-order tensor bytes. Their filename
and parent directories identify the original source, identity and split.
`manifest.json` records shape, dtype, full profile, original dataset path, source
and tensor SHA-256, bounding boxes, keypoints, confidence and exclusions.

```python
import numpy as np

tensor = np.fromfile(
    "artifacts/arcface/enrolment/George_W_Bush/George_W_Bush_0001.jpg.bin",
    dtype="<f4",
).reshape(1, 3, 112, 112)
```

Use `uint8`/`int8` and the manifest shape for integer profiles. Preparation refuses
existing output directories. A failed preparation can leave partial binaries
but **no completed manifest**; inspect/remove that specific failed output or use a
new directory before retrying. Do not evaluate or edit a partial run.

## 5. Evaluate a recognition model

Place your licensed model under the git-ignored `models/` directory.

```bash
python evaluate.py \
  --manifest artifacts/arcface/manifest.json \
  --model models/arcface.onnx \
  --metric cosine --threshold 0.5 \
  --warmup 3 --threads 1 \
  --output reports/arcface-cosine
```

For TensorFlow Lite, prepare tensors with the matching profile, then:

```bash
python evaluate.py \
  --manifest artifacts/mobilefacenet/manifest.json \
  --model models/mobilefacenet.tflite \
  --metric euclidean --threshold 1.0 \
  --output reports/mobilefacenet-euclidean
```

### GPU inference

```bash
# ONNX on CUDA (GPU devcontainer); --gpu-id selects the device
python evaluate.py --device gpu --gpu-id 0 \
  --manifest artifacts/arcface/manifest.json --model models/arcface.onnx \
  --metric cosine --threshold 0.3 --output reports/arcface-gpu

# TFLite with a GPU delegate shared library
python evaluate.py --device gpu --gpu-delegate /path/to/libtensorflowlite_gpu_delegate.so \
  --manifest artifacts/mobilefacenet/manifest.json --model models/mobilefacenet.tflite \
  --metric cosine --threshold 0.3 --output reports/mobilefacenet-gpu
```

ONNX GPU runs fail if `CUDAExecutionProvider` is unavailable or fails to
initialise (the report records the active providers). ONNX Runtime may still
place individual unsupported operators on CPU, which is standard behaviour.
The TensorFlow pip wheel does not bundle a TFLite GPU delegate on Linux; build
`//tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so` with Bazel (or
use a vendor delegate) and pass it with `--gpu-delegate` or the
`TFLITE_GPU_DELEGATE` environment variable. Without a delegate,
`--device gpu` on a `.tflite` model is an error.

Both thresholds above are **illustrative, not recommended operating points**.
Choose a threshold on a separate calibration/development dataset before
evaluation. Do not tune it against the reported test outcomes. If datasets
overlap or the model was trained on LFW, report that limitation.

The adapters enforce a single model input, configured shape/dtype, and a single
embedding output. Dynamic input dimensions are resolved to the configured shape.
For multi-output models, pass `--output-name EXACT_NAME`. Embeddings must have
one non-singleton vector dimension, be finite and nonzero; dimension changes
between images are rejected. ONNX embedding outputs must be floating point
(export a dequantized output). TFLite per-tensor integer input/output quantization
is supported, and input scale/zero point must match the profile; per-axis
quantization is rejected.

Manifest tensor checksums, byte counts and split membership are validated before
inference. Reusing source paths/tensor paths or identical source content across
enrolment/test is rejected. This detects byte-identical copies, **not** near-duplicate
photos or identities mislabeled in a dataset.

### Protocol and distance metrics

1. Compute one embedding per selected image.
2. By default L2-normalize every embedding.
3. Average the enrolment embeddings per identity and L2-normalize that template.
4. Compare **each test embedding with every identity template**:
   its own identity is a genuine comparison; every other identity is an impostor.
5. Apply the explicitly supplied threshold to each comparison.
6. Also select the best template for closed-set rank-1 identification.
   Exact ties go to the alphabetically first identity.

`--no-normalize-embeddings` disables both embedding and template L2 normalization;
templates remain arithmetic means. Cosine similarity still normalizes internally
by definition. Without this option cosine and dot product are equivalent on
unit-normalized vectors. There are no assumptions about embedding dimension.

| CLI metric | Definition | Accepted when |
|---|---|---|
| `cosine` | `dot(a,b) / (norm(a)*norm(b))` | score >= threshold |
| `dot` | `sum(a*b)` | score >= threshold |
| `euclidean` | `sqrt(sum((a-b)**2))` | distance <= threshold |
| `manhattan` | `sum(abs(a-b))` | distance <= threshold |

With K identities and Y probes per identity, there are K*Y genuine and
K*Y*(K-1) impostor comparisons. This imbalance makes raw accuracy potentially
misleading; also compare balanced accuracy, FAR and FRR.

### Outputs and definitions

Each new report directory contains:

- `report.json`: configuration, model/manifest hashes, runtime/environment,
  counts, threshold, metrics, ROC arrays and timing.
- `scores.csv`: one row per probe/claimed-identity comparison, with raw score,
  genuine label, accept decision and rank-1 identity.
- `embeddings.npz`: embeddings in manifest-image order, templates with ordered
  identity labels, sources, splits and per-image measured milliseconds.
  Embeddings/templates are normalized if normalization was enabled.

Reports refuse existing destinations. The CLI prints a summary as well.

| Report term | Definition |
|---|---|
| True accepted (TA / TP) | Genuine comparison accepted |
| False rejected (FR / FN) | Genuine comparison rejected |
| False accepted (FA / FP) | Impostor comparison accepted |
| True rejected (TR / TN) | Impostor comparison rejected |
| FAR / FMR | `FA / (FA + TR)` |
| FRR / FNMR | `FR / (FR + TA)` |
| TAR / recall / sensitivity | `TA / (TA + FR)` |
| TRR / specificity | `TR / (TR + FA)` |
| Accuracy | `(TA + TR) / all comparisons` |
| Balanced accuracy | `(TAR + TRR) / 2` |
| Precision | `TA / (TA + FA)`; zero if no accepts |
| F1 | Harmonic mean of precision and recall; zero if both zero |
| ROC AUC | Trapezoidal area under the verification ROC, ties grouped |
| Interpolated EER | Linear ROC interpolation where FAR = FRR |
| Rank-1 accuracy | Fraction of probes whose best template is their identity |
| Rank-1 correct-and-accepted rate | Fraction of probes both correctly ranked and accepted at threshold |

ROC AUC and EER describe the test-score distribution, **not** an independently
validated threshold. Correlated comparisons from the same subjects are not
independent trials; no confidence intervals are claimed. All probes belong to
enrolled identities, so these reports **do not measure open-set unknown-person
rejection**. Face detection failures stop preparation and are not counted as
recognition false rejections; report any preparation exclusions separately.

### Timing and fair comparisons

The timed operation is a synchronous `runtime.embed(tensor)` call including
runtime input/output transfer, output dequantization and vector validation.
It excludes image decoding/detection/alignment, tensor disk reads, model loading,
warmups, embedding normalization and template/score construction.

Reports include mean, median, p95, minimum and maximum milliseconds across all
selected images, plus enrolment/test means and counts. A default three warmup
calls use the first image and are not included. This is batch-1 embedding latency,
not whole-pipeline latency or sustained throughput.

For model comparisons, keep image selection, detector settings, hardware,
thread count and system load fixed. Change only preprocessing fields required
by each model. Avoid concurrent benchmarks. Record CPU model, Docker limits,
power mode and detector weight checksum alongside the report; platform metadata
alone may not expose the physical CPU under WSL2. The test suite is not streaming:
it retains embeddings and all comparison rows, so memory/output grows with
`number_of_probes * number_of_identities`.

### Example results (verified on this repository)

InsightFace `buffalo_sc/w600k_mbf.onnx` (MobileFaceNet) and
`buffalo_l/w600k_r50.onnx` (ArcFace ResNet-50) with
[`configs/arcface_112_nchw.json`](configs/arcface_112_nchw.json), LFW original,
`--max-identities 40 --enrolment 3 --testing 5` (200 genuine and 7,800 impostor
comparisons), cosine threshold 0.3, WSL2 + RTX A2000 Laptop GPU:

| Model | Device | TA | FR | FA | TR | Mean embed ms |
|---|---|---|---|---|---|---|
| w600k_mbf | CPU (4 threads) | 199 | 1 | 0 | 7800 | ~8 |
| w600k_mbf | GPU (CUDA) | 199 | 1 | 0 | 7800 | ~2 |
| w600k_r50 | GPU (CUDA) | 200 | 0 | 0 | 7800 | ~6.6 |

These insightface models were trained on WebFace600K, which can overlap with LFW
identities, and this subset is tiny; treat the numbers as a smoke test, not a
benchmark. As a sanity check, deliberately preprocessing with `"color": "BGR"`
degraded MobileFaceNet to 2 false rejections and a lower ROC AUC. Model weights
are not included; check insightface's licence before use.

## 6. Automated tests and checks

Inside the devcontainer:

```bash
python -m pytest tests -q
python -m ruff check .
python -m compileall -q face_eval download_dataset.py prepare_tensors.py evaluate.py
python -m pip check
```

Or from the WSL host:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/workspaces/lfw-model-eval" \
  lfw-model-eval:cpu python -m pytest tests -q
```

Tests use generated BMP/JPEG images, injected detector results and locally
generated embedding models. They cover safe archive handling/checksums, sorted
disjoint selection, transform recovery, sizes/layouts/color/normalization,
integer quantization, binary round trips, manifest corruption/leakage,
threshold directions and exact confusion counts/ROC ties. Integration tests
execute real ONNX Runtime models with 128/192/512 outputs and real float/int8
TFLite models. They do not need LFW, pretrained recognition weights or RetinaFace
weight downloads, and do not establish real-world recognition accuracy.

All apps support `--help`. Errors surface as explicit exceptions/nonzero exits;
there are no skipped faces, silent runtime fallback or fabricated success reports.

## Project layout and extension points

```text
.devcontainer/           CPU Docker build and VS Code setup
.devcontainer/gpu/       CUDA/cuDNN + onnxruntime-gpu variant
configs/                 Example model preprocessing contracts
face_eval/config.py      Profile validation and target landmarks
face_eval/datasets.py    Download/extraction and deterministic selection
face_eval/preprocessing.py  Shared detector/alignment/tensor entry point
face_eval/runtimes.py    ONNX/TFLite CPU/GPU adapters
face_eval/metrics.py     Scores, confusion counts, ROC/AUC/EER
face_eval/evaluation.py  Manifest validation, templates and report persistence
download_dataset.py     Download CLI
prepare_tensors.py      Tensor preparation CLI
evaluate.py             Recognition evaluation CLI
tests/                  Offline unit and runtime integration tests
```

Add profiles for new models rather than model-name conditionals in the pipeline.
New datasets normally need only an identity-folder adapter or archive URL.
New accelerators (e.g. TensorRT, OpenVINO, NNAPI) belong in `face_eval/runtimes.py`:
select the provider/delegate explicitly, verify the actual execution device and
never silently use CPU under a GPU-labelled run.

## Data, licensing and responsible use

LFW contains identifiable people and has demographic/collection biases. Obtain
appropriate permission and review the dataset and each model's terms before use.
See the [LFW project](https://vis-www.cs.umass.edu/lfw/) and
[scikit-learn's LFW loader metadata](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_lfw.py)
for dataset provenance and the mirror checksums.

Treat images, tensor files, embeddings and detailed comparison reports as
sensitive biometric data. `data/`, `models/`, `artifacts/` and `reports/` are
git-ignored; that is not encryption or access control. Do not commit or share
them without authorization. The suite includes no recognition model weights or
dataset images and makes no claims about their licensing, fairness or suitability
for deployment.
