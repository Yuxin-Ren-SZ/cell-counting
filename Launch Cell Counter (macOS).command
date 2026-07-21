#!/bin/bash
# Double-click in Finder to launch the napari Cell Counter plugin.
# (If macOS blocks it the first time: right-click -> Open, or run
#  `chmod +x` on this file.)

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE" || exit 1
ENV_NAME="cell-counting"

echo "Cell Counter launcher"
echo "Repo: $HERE"

# --- locate a conda installation ---------------------------------------
find_conda_base() {
    for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
                "$HOME/mambaforge" "/opt/homebrew/Caskroom/miniconda/base" \
                "/opt/miniconda3" "/opt/anaconda3" "/usr/local/anaconda3"; do
        if [ -f "$base/etc/profile.d/conda.sh" ]; then
            echo "$base"; return 0
        fi
    done
    if command -v conda >/dev/null 2>&1; then
        conda info --base; return 0
    fi
    return 1
}

CONDA_BASE="$(find_conda_base)"
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo
    echo "Could not find conda. Install Miniconda from"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo "then create the environment:"
    echo "  conda env create -f \"$HERE/environment.yml\""
    echo
    read -r -p "Press Return to close."
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda activate "$ENV_NAME" 2>/dev/null; then
    echo
    echo "Conda environment '$ENV_NAME' not found. Create it with:"
    echo "  conda env create -f \"$HERE/environment.yml\""
    echo
    read -r -p "Press Return to close."
    exit 1
fi

echo "Activated conda env: $ENV_NAME"
echo "Starting napari..."

# --- launch ------------------------------------------------------------
if command -v napari-cell-counter >/dev/null 2>&1; then
    napari-cell-counter
else
    python -m napari_cell_counter
fi

status=$?
if [ $status -ne 0 ]; then
    echo
    echo "napari exited with an error (code $status)."
    read -r -p "Press Return to close."
fi
