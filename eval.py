"""Backward-compatible entrypoint for model evaluation."""

import runpy

if __name__ == "__main__":
    runpy.run_module("scripts.eval", run_name="__main__")
