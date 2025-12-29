#!/usr/bin/env python
import argparse
from pathlib import Path
from src.engine.runner import main

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", type=str, default=str(Path("configs") / "small.yaml"),
                    help="Path to YAML config")
    args = ap.parse_args()
    main(args.config)

if __name__ == "__main__":
    cli()