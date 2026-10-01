"""Backward-compatible entrypoint for Darcy rough experiment."""

import runpy

if __name__ == "__main__":
    runpy.run_module("experiments.ex_darcyrough", run_name="__main__")
