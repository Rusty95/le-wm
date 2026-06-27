#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${LEWM_ENV_DIR:-${ROOT_DIR}/.venvs/lewm}"
PYTHON_BIN="${PYTHON_BIN:-}"
USE_CONDA=0

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
    echo "python3.10 not found; falling back to python3.11 for the LeWM environment."
  elif command -v conda >/dev/null 2>&1; then
    USE_CONDA=1
  else
    echo "Neither python3.10/python3.11 nor conda was found. Set PYTHON_BIN=/path/to/python." >&2
    exit 1
  fi
fi

echo "Creating LeWM environment at: ${ENV_DIR}"
if [[ "${USE_CONDA}" == "1" ]]; then
  echo "Using conda fallback with Python 3.10."
  if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    conda create -y -p "${ENV_DIR}" python=3.10 pip
  else
    echo "Environment already exists; reusing it."
  fi
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_DIR}"
elif command -v uv >/dev/null 2>&1; then
  echo "Using uv with Python: ${PYTHON_BIN}"
  if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    uv venv --python "${PYTHON_BIN}" "${ENV_DIR}"
  else
    echo "Environment already exists; reusing it."
  fi
  # shellcheck source=/dev/null
  source "${ENV_DIR}/bin/activate"
else
  echo "uv not found; using stdlib venv with Python: ${PYTHON_BIN}"
  if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${ENV_DIR}"
  else
    echo "Environment already exists; reusing it."
  fi
  # shellcheck source=/dev/null
  source "${ENV_DIR}/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel

if command -v uv >/dev/null 2>&1; then
  PIP_INSTALL=(uv pip install)
else
  PIP_INSTALL=(python -m pip install)
fi

echo "Installing local stable-pretraining and stable-worldmodel packages."
"${PIP_INSTALL[@]}" -e "${ROOT_DIR}/stable-pretraining"
"${PIP_INSTALL[@]}" -e "${ROOT_DIR}/stable-worldmodel" --no-deps

echo "Installing LeWM runtime helpers."
"${PIP_INSTALL[@]}" \
  hydra-core omegaconf lightning wandb transformers einops h5py hdf5plugin \
  numpy torch torchvision lancedb pylance pyarrow pillow tqdm typer rich \
  gymnasium loguru tabulate "imageio[ffmpeg]"

cat <<EOF

LeWM environment is ready.

Activate it with:
  source "${ROOT_DIR}/activate_lewm.sh"

Recommended data/cache variables:
  export STABLEWM_HOME="\${STABLEWM_HOME:-${ROOT_DIR}/.stable-wm}"
  export LOCAL_DATASET_DIR="\${LOCAL_DATASET_DIR:-\$STABLEWM_HOME}"
  export SPT_CACHE_DIR="\${SPT_CACHE_DIR:-${ROOT_DIR}/.stable-pretraining}"

Run the smoke test:
  python "${ROOT_DIR}/le-wm/scripts/smoke_test_lewm.py"
EOF
