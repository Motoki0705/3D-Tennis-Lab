# Install / update uv

```powershell
# Install (official script)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Make uv available in the current session (if PATH not updated yet)
$env:Path = "$HOME\.local\bin;$env:Path"

# Verify / update
uv --version
uv self update
```

# Virtual environment & Python

```powershell
# Create .venv (uses your default Python)
uv venv

# Create with a specific Python
uv python install 3.11.9
uv venv --python 3.11

# Activate / deactivate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
deactivate
```

# Lock & install dependencies

```powershell
# Create/refresh the lockfile
uv lock

# Install base deps (+ default dev group)
uv sync

# Install ALL groups (base + every group you defined)
uv sync --all-groups
# or list groups explicitly (repeatable)
uv sync --group ml --group vision --group data --group notebook --group docs

# Exclude default groups
uv sync --no-default-groups --group ml --group vision

# Only one specific group (without defaults)
uv sync --only-group docs
```

# Add / remove packages (writes to pyproject + lock)

```powershell
# Base (runtime) dependency
uv add "requests>=2.32"

# In a specific group
uv add --group ml "pytorch-lightning>=2.5"

# From the CUDA index (assuming you declared torch-cuda in pyproject)
uv add --group ml "torch==2.8.0+cu128" --index torch-cuda
uv add --group ml "torchvision==0.23.0+cu128" --index torch-cuda
uv add --group ml "torchaudio==2.8.0+cu128" --index torch-cuda

# Remove
uv remove --group vision smplx
```

# Git dependency (if you prefer commands instead of editing pyproject)

```powershell
uv add --group vision "panopticapi@git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0"
```

# Run commands inside the environment

```powershell
# uv auto-syncs if needed, then runs
uv run python -c "import requests; print(requests.__version__)"
uv run pytest -q
```

# CUDA quick check

```powershell
uv run python - << 'PY'
import torch, torchvision, torchaudio
print("torch:", torch.__version__)
print("cuda is available:", torch.cuda.is_available())
PY
```

# Useful maintenance

```powershell
# See resolved dependency tree
uv tree

# Clean caches (if you need to free space)
uv cache clean

# Help
uv --help
uv sync --help
```

> Tip: If a command like `--all-groups` isn’t recognized, update uv (`uv self update`) or use the explicit repeated `--group` flags shown above.
