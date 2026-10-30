"""Shared paths and derived settings for the experiment entry points."""

import os


SRC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_PATH = os.path.join(SRC_ROOT, "data")
MODEL_PATH = os.path.join(SRC_ROOT, "models")


def set_attention_resolutions(config):
    """Derive the input and attention grid sizes from the sampling settings."""
    config["res_input"] = int(
        (config["resolution_datasets"] - 1) / config["subsample_nodes"] + 1
    )
    config["res_att"] = int(
        (config["res_input"] - config["patch_size"] + 2 * config["patch_padding"])
        / config["subsample_attn"]
        + 1
    )


def set_output_names(config, model_name):
    """Set the checkpoint and associated result filenames."""
    config["model_save_path"] = MODEL_PATH
    config["model_name"] = model_name
    config["result_name"] = model_name[:-3] + ".pkl"
    config["mat_name"] = model_name[:-3] + ".mat"
