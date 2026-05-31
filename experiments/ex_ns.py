import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hano.trainer import train_NS_model

SRC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_PATH = os.path.join(SRC_ROOT, "data")
MODEL_PATH = os.path.join(SRC_ROOT, "models")


if __name__ == "__main__":
    r_dic = {}
    r_dic["train_path"] = "NavierStokes_V1e-5_N1200_T20.mat"
    r_dic["test_path"] = "NavierStokes_V1e-5_N1200_T20.mat"
    r_dic["Data_path"] = DATA_PATH
    r_dic["boundary_condition"] = None
    r_dic["train_len"] = 1000
    r_dic["test_len"] = 200
    r_dic["resolution_datasets"] = 64
    r_dic["batch_size"] = 20
    r_dic["T"] = 10
    r_dic["T_in"] = 10
    r_dic["in_dim"] = r_dic["T_in"]

    r_dic["xGN"] = False
    r_dic["subsample_nodes"] = 1
    r_dic["subsample_attn"] = 1
    r_dic["patch_padding"] = 1
    r_dic["patch_size"] = 3
    r_dic["res_input"] = int((r_dic["resolution_datasets"] - 1) / r_dic["subsample_nodes"] + 1)
    r_dic["res_att"] = int((r_dic["res_input"] - r_dic["patch_size"] + 2 * r_dic["patch_padding"]) / r_dic["subsample_attn"] + 1)
    r_dic["res_output"] = 64

    # HANO multigrid-attention backbone.
    r_dic["epochs"] = 500
    r_dic["feature_dim"] = 18
    r_dic["num_layer"] = 1
    r_dic["num_iteration"] = [[1, 0], [1, 0], [1, 0]]
    r_dic["padding_mode"] = "circular"
    r_dic["activation"] = "gelu"
    r_dic["last_layer"] = "conv"

    r_dic["model_save_path"] = MODEL_PATH
    r_dic["model_name"] = "NS_1e-5_N1200.pt"
    r_dic["result_name"] = str(r_dic["model_name"][0:-3]) + ".pkl"
    r_dic["mat_name"] = str(r_dic["model_name"][0:-3]) + ".mat"

    print(r_dic)
    train_NS_model(r_dic)

    print("END")
