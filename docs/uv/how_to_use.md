# uv Usage Guidelines

---

## 1. Activating Environments

- **Windows (PowerShell)**

  ```powershell
  .\.venv\Scripts\Activate.ps1
  deactivate   # exit
  ```

- **WSL (bash/zsh)**

  ```bash
  source .venv-wsl/bin/activate
  deactivate
  ```

---

## 2. Using `uv run` Without Activation

`uv run` executes commands inside the project’s environment **without manual activation**.

- **Windows (defaults to `.venv`)**

  ```powershell
  uv run python app.py
  uv run pytest -q
  ```

- **WSL**

  - To always use `.venv-wsl`:

    ```bash
    export UV_PROJECT_ENVIRONMENT=".venv-wsl"
    uv run python app.py
    ```

  - To use the currently active venv explicitly:

    ```bash
    source .venv-wsl/bin/activate
    uv run --active python app.py
    ```
