# Step 5.2: Image saving utility
import cv2
from pathlib import Path


def save_once(images_dir: Path, file_name: str, frame_bgr):
    """Saves a frame as a JPEG image with provided file_name, creating the directory if needed."""
    images_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(images_dir / file_name), frame_bgr)
    return file_name
