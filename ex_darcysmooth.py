"""Backward-compatible entrypoint for Darcy smooth experiment."""

import runpy

if __name__ == "__main__":
    runpy.run_module("experiments.ex_darcysmooth", run_name="__main__")
