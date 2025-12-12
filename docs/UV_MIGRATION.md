# Migration to UV Package Manager

## What Changed

PIVOT has migrated from `pip` + `requirements.txt` to **`uv`** for package management. This provides:

- ⚡ **10-100x faster** installation and dependency resolution
- 🔒 **Deterministic builds** with `uv.lock`
- 🎯 **Better dependency resolution** (no more conflicts)
- 🐍 **Automatic Python version management**
- 🔄 **Workspace support** for monorepos

## Quick Migration Guide

### For Contributors

**Old way (pip):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

**New way (uv):**
```bash
uv sync --dev
```

That's it! uv handles everything: virtual environment, dependencies, and installation.

### Common Commands

| Task | Old (pip) | New (uv) |
|------|-----------|----------|
| Install deps | `pip install -r requirements.txt` | `uv sync` |
| Install dev deps | `pip install -r requirements-dev.txt` | `uv sync --dev` |
| Add package | `pip install package && pip freeze > requirements.txt` | `uv add package` |
| Add dev package | `pip install package` (manual) | `uv add --dev package` |
| Run command | `python script.py` | `uv run python script.py` |
| Run tests | `pytest` | `uv run pytest` |

### GPU Installation

**NVIDIA (CUDA):**
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv sync
```

**AMD (ROCm):**
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
uv sync
```

**Intel (OneAPI):**
```bash
uv pip install torch torchvision intel-extension-for-pytorch \
  --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
uv sync
```

**CPU Only (default):**
```bash
uv sync
```

## Key Files

- **`pyproject.toml`**: Project metadata and dependencies (single source of truth)
- **`uv.lock`**: Locked dependencies (commit this!)
- **`.python-version`**: Python version (3.13)
- **`requirements*.txt`**: Legacy files (Docker only, not used locally)

## Docker

Docker builds still use `requirements.txt` files but install via `uv` inside containers. No changes needed to Docker workflow.

## Troubleshooting

### "Command not found: uv"
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### "No interpreter found for Python X.X"
```bash
uv python install 3.13
```

### "Failed to build pivot"
Make sure you're in the project root and `pyproject.toml` exists.

## Benefits for PIVOT

1. **Faster CI/CD**: Dependencies install in seconds instead of minutes
2. **No version conflicts**: uv's resolver is smarter than pip's
3. **Reproducible builds**: `uv.lock` ensures everyone gets the same versions
4. **Simpler commands**: One command to rule them all (`uv sync`)
5. **GPU flexibility**: Easy switching between CUDA/ROCm/Intel/CPU

## For Docker Users

No changes needed! Continue using:
```bash
make docker-build
make docker-train
```

Docker images use uv internally for faster builds.

## Learn More

- [uv Documentation](https://docs.astral.sh/uv/)
- [Migration Guide](https://docs.astral.sh/uv/guides/migration/)
- [Project Configuration](https://docs.astral.sh/uv/concepts/projects/)
