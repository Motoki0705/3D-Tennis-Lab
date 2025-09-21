import logging
import sys
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig

import os

# Get the absolute path of the directory containing run.py
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the WASB-SBDT/src directory
wasb_src_path = os.path.join(script_dir, "..", "..", "..", "third_party", "WASB-SBDT", "src")
# Add it to the Python path
sys.path.append(os.path.normpath(wasb_src_path))

from runners import select_runner
from utils import mkdir_if_missing

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name="detect", config_path="configs")
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))

    if cfg.get("output_dir") is None:
        cfg["output_dir"] = HydraConfig.get().run.dir
    mkdir_if_missing(cfg["output_dir"])

    # Select and run the runner
    runner = select_runner(cfg)

    # Get video_path and output_path from config
    # These can be set via command line, e.g.:
    # python run.py video_path=my_video.mp4 output_path=out.mp4
    video_path = cfg.get("video_path")
    output_path = cfg.get("output_path")

    if not video_path or not output_path:
        log.error("Please provide 'video_path' and 'output_path' as command-line arguments.")
        return

    runner.run(video_path, output_path)


if __name__ == "__main__":
    main()
