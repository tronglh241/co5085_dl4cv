# Development Guide

Day-to-day commands. For environment setup, see [SETUP.md](SETUP.md).

## Contents

- [Docker](#docker)
  - [1. Run experiments](#1-run-experiments)
  - [2. Preview GitHub Pages](#2-preview-github-pages)
  - [3. Build the report](#3-build-the-report)
  - [4. Lint](#4-lint)
- [Manual](#manual)
  - [1. Run experiments](#1-run-experiments-1)
  - [2. Lint](#2-lint)

## Docker

### 1. Run experiments

**1.1 Start a shell**

Opens a shell in the experiment container, with the repo mounted at `/workspace`. Pass `gpu` to use the GPU or `cpu` if you have no NVIDIA stack.

```bash
bash scripts/shell/dev.sh <gpu|cpu>
```

Examples:

```bash
bash scripts/shell/dev.sh gpu
# or
bash scripts/shell/dev.sh cpu
```

**1.2 Train a model**

Trains a model as described by a YAML config file.

```bash
python scripts/train.py <config_path>
```

Example:

```bash
python scripts/train.py configs/e1/train_softmax.yml
```

**1.3 Evaluate a model**

Evaluates a trained model as described by a YAML config file. The checkpoint to load is set in the config under `loader.path`.

```bash
python scripts/evaluate.py <config_path>
```

Example:

```bash
python scripts/evaluate.py configs/e1/test_softmax.yml
```

**1.4 Start TensorBoard**

Serves the training logs from `logs/` at <http://localhost:6006>.

```bash
bash scripts/tensorboard/serve.sh
```

**1.5 Export TensorBoard logs to CSV**

Extracts the TensorBoard logs into a CSV file. The output `metrics.csv` is written to the same TensorBoard log directory the logs are read from.

```bash
python scripts/tensorboard/extract.py
```

**1.6 Download and verify artifacts**

Downloads the shared checkpoints and training logs, then verifies them against the checksum manifests in `checksums/`. The manifests cover checkpoint `.pt`/`.yml` files and TensorBoard event files.

```bash
bash scripts/artifacts/download.sh
bash scripts/artifacts/checksums.sh
```

`checksums.sh` prints `all artifacts verified` when every file matches; otherwise it lists the missing or mismatching files and exits non-zero.

Maintainers regenerate the manifests after changing the shared assets, and commit the result:

```bash
bash scripts/artifacts/checksums.sh generate
```

**1.7 Generate report assets**

Generates the figures and CSV values used by the report. Start the experiment container and run the generator inside it:

```bash
bash scripts/generate.sh all
```

Checkpoints and logs must already be present, and each test config must point to its best checkpoint. Figures go to `reports/images/` and values to `reports/results/`.

### 2. Preview GitHub Pages

**2.1 Start a shell**

Opens a shell in the `gpu` or `cpu` container; both include the Jekyll toolchain.

```bash
bash scripts/shell/dev.sh <gpu|cpu>
```

Example:

```bash
bash scripts/shell/dev.sh gpu
# or
bash scripts/shell/dev.sh cpu
```

**2.2 Serve the site**

Builds the site and serves a live preview at <http://localhost:4000>.

```bash
bash scripts/page/serve.sh
```

### 3. Build the report

**3.1 Start a shell**

Opens a shell in the `latex` container.

```bash
bash scripts/shell/latex.sh
```

**3.2 Build the PDF**

Compiles the reports into PDFs.

```bash
bash scripts/report/build.sh
```

### 4. Lint

**4.1 Start a shell**

Opens a shell in the experiment container. Skip this if you already have one open from [Run experiments](#1-run-experiments).

```bash
bash scripts/shell/dev.sh <gpu|cpu>
```

**4.2 Run lint**

Checks code style with `isort` and `flake8`. Run it after every code change.

```bash
bash scripts/lint.sh
```

## Manual

Experiments and lint only. The site and report require Docker.

### 1. Run experiments

**1.1 Activate the Python environment**

Uses the environment created in [SETUP.md](SETUP.md). No container shell is needed.

```bash
conda activate co5085
```

**1.2 Train a model**

Trains a model as described by a YAML config file.

```bash
python scripts/train.py <config_path>
```

Example:

```bash
python scripts/train.py configs/e1/train_softmax.yml
```

**1.3 Evaluate a model**

Evaluates a trained model as described by a YAML config file. The checkpoint to load is set in the config under `loader.path`.

```bash
python scripts/evaluate.py <config_path>
```

Example:

```bash
python scripts/evaluate.py configs/e1/test_softmax.yml
```

**1.4 Start TensorBoard**

Serves the training logs from `logs/` at <http://localhost:6006>.

```bash
bash scripts/tensorboard/serve.sh
```

**1.5 Export TensorBoard logs to CSV**

Extracts the TensorBoard logs into a CSV file. The output `metrics.csv` is written to the same TensorBoard log directory the logs are read from.

```bash
python scripts/tensorboard/extract.py
```

### 2. Lint

Checks code style with `isort` and `flake8`. Run it after every code change.

```bash
bash scripts/lint.sh
```
