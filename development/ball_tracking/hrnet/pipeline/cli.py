from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config
from .pipeline import AnnotationPipeline


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s][%(levelname)s] %(name)s - %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])

    parser = argparse.ArgumentParser(prog="python -m pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="scan videos, run inference, and export clips")

    accept_parser = sub.add_parser("accept", help="mark a clip as accepted")
    accept_parser.add_argument("clip", help="clip reference, e.g. game1/Clip1")

    reject_parser = sub.add_parser("reject", help="mark a clip as rejected")
    reject_parser.add_argument("clip", help="clip reference")

    sub.add_parser("finalize", help="merge accepted clips into final annotations")

    args, hydra_overrides = parser.parse_known_args(argv)
    cfg = load_config(hydra_overrides)

    setup_logging(cfg.logging.level)

    pipeline = AnnotationPipeline(cfg)

    if args.command == "run":
        pipeline.run()
    elif args.command == "accept":
        pipeline.accept(args.clip)
    elif args.command == "reject":
        pipeline.reject(args.clip)
    elif args.command == "finalize":
        pipeline.finalize()
    else:  # pragma: no cover - safeguard
        parser.error(f"Unsupported command {args.command}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
