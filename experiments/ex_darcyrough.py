import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hano.trainer import train_model

SRC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_PATH = os.path.join(SRC_ROOT, "data")
MODEL_PATH = os.path.join(SRC_ROOT, "models")


if __name__ == "__main__":
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
    r_dic["res_input"] = int((r_dic["resolution_datasets"] - 1) / r_dic["subsample_nodes"] + 1)
    r_dic["res_att"] = int((r_dic["res_input"] - r_dic["patch_size"] + 2 * r_dic["patch_padding"]) / r_dic["subsample_attn"] + 1)
    r_dic["res_output"] = 256

    r_dic["epochs"] = 500
    r_dic["feature_dim"] = 64
    r_dic["window_size"] = [4, 4, 4]
    r_dic["depths"] = [1, 1, 1]
    r_dic["num_heads"] = [1, 1, 1]

    r_dic["F_modes"] = 12
    r_dic["F_width"] = 64
    r_dic["num_spectral_layers"] = 5
    r_dic["mlp_hidden_dim"] = 128
    r_dic["F_padding"] = 5

    r_dic["model_save_path"] = MODEL_PATH
    r_dic["model_name"] = "darcyrough_res256.pt"
    r_dic["result_name"] = str(r_dic["model_name"][0:-3]) + ".pkl"
    r_dic["mat_name"] = str(r_dic["model_name"][0:-3]) + ".mat"

    print(r_dic)
    train_model(r_dic)

    print("END")
