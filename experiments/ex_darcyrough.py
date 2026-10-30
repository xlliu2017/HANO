import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hano.trainer import train_model
from experiments.common import (
    DATA_PATH, MODEL_PATH, SRC_ROOT, set_attention_resolutions, set_output_names,
)


def build_config():
    r_dic = {}

    r_dic["model"] = "HANO"

    r_dic["train_path"] = "darcy_rough_train.mat"
    r_dic["val_path"] = "darcy_rough_val.mat"
    r_dic["test_path"] = "darcy_rough_test.mat"
    r_dic["Data_path"] = DATA_PATH
    r_dic["train_len"] = 1280
    r_dic["val_len"] = 112
    r_dic["test_len"] = 112
    r_dic["resolution_datasets"] = 512
    r_dic["batch_size"] = 8

    r_dic["boundary_condition"] = "dirichlet"
    r_dic["losstype"] = "H1"

    r_dic["xGN"] = True
    r_dic["subsample_nodes"] = 1
    r_dic["subsample_attn"] = 2
    r_dic["patch_padding"] = 1
    r_dic["patch_size"] = 4
    set_attention_resolutions(r_dic)
    r_dic["res_output"] = 256

    # HANO multigrid-attention backbone.
    r_dic["epochs"] = 500
    r_dic["feature_dim"] = 64
    r_dic["num_layers"] = 1
    r_dic["num_iterations"] = [[1, 0], [1, 0], [1, 0]]
    r_dic["padding_mode"] = "zeros"
    r_dic["activation"] = "gelu"
    r_dic["last_layer"] = "conv"

    set_output_names(r_dic, "darcyrough_res256.pt")
    return r_dic


if __name__ == "__main__":
    r_dic = build_config()
    print(r_dic)
    train_model(r_dic)

    print("END")
