# Step 8.1: Bootstrap tasks like logging setup (can be expanded later)
import logging
from pathlib import Path


def setup_logging(level: str = "INFO", file_path: str | None = None):
    """Configure console and optional file logging.

    If file_path is provided, ensure the parent directory exists and write INFO logs there.
    """
    # Reset basic config
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if file_path:
        try:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(p, encoding="utf-8")
            fh.setLevel(getattr(logging, level.upper(), logging.INFO))
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logging.getLogger().addHandler(fh)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to add file logger at {file_path}: {e}")
