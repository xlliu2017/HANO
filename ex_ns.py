"""Backward-compatible entrypoint for Navier-Stokes experiment."""

import runpy

if __name__ == "__main__":
    runpy.run_module("experiments.ex_ns", run_name="__main__")
