"""Backward-compatible entrypoint for FNO multiscale experiment."""

import runpy

if __name__ == "__main__":
    runpy.run_module("experiments.ex_fno_multiscale", run_name="__main__")
