#!/bin/bash
# ══════════════════════════════════════════════════════════════
# PostAIMSPy Installer
# ══════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           PostAIMSPy v1.0.0 — Installer                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

if [ ! -f "$SCRIPT_DIR/pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found in $SCRIPT_DIR"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/postaimspy/__init__.py" ]; then
    echo "ERROR: postaimspy/__init__.py not found."
    echo ""
    echo "  Expected layout:"
    echo "    project_folder/"
    echo "    ├── install.sh"
    echo "    ├── pyproject.toml"
    echo "    └── postaimspy/"
    echo "        ├── __init__.py"
    echo "        ├── cli.py"
    echo "        └── ..."
    exit 1
fi

if [ -n "$1" ]; then
    echo "Activating environment: $1"
    source "$1/bin/activate" 2>/dev/null || echo "WARNING: Could not activate $1"
fi

PYTHON=$(which python3 2>/dev/null || which python)
echo "Python  : $PYTHON"
echo "Version : $($PYTHON --version 2>&1)"
echo ""

echo "Cleaning previous builds..."
rm -rf "$SCRIPT_DIR/build" "$SCRIPT_DIR/dist" "$SCRIPT_DIR"/*.egg-info
pip uninstall postaimspy aimspy -y 2>/dev/null || true

echo ""
echo "Installing postaimspy..."
cd "$SCRIPT_DIR"
pip install .

echo ""
echo "Installing optional dependencies (mdtraj, matplotlib)..."
pip install mdtraj matplotlib 2>/dev/null || echo "WARNING: Optional deps failed. Core still works."

echo ""
echo "Verifying installation..."
$PYTHON -c "import postaimspy; print(f'  postaimspy {postaimspy.__version__} imported OK')" || {
    echo "ERROR: import failed!"
    exit 1
}

postaimspy --version 2>/dev/null && echo "  postaimspy CLI works OK" || {
    echo "  NOTE: 'postaimspy' command not on PATH."
    echo "  Use instead: python -m postaimspy --help"
}

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  Installation complete!"
echo ""
echo "  Quick start:"
echo "    postaimspy --help"
echo "    postaimspy init                     # generate input.yaml"
echo "    postaimspy run input.yaml           # run full pipeline"
echo "    postaimspy run input.yaml --steps separate"
echo "    postaimspy run input.yaml --steps cluster"
echo "    postaimspy run input.yaml --steps align"
echo "══════════════════════════════════════════════════════════"
