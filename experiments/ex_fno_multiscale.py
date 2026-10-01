import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hano.trainer import train_model

SRC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_PATH = os.path.join(SRC_ROOT, "data")
MODEL_PATH = os.path.join(SRC_ROOT, "models")


if __name__ == "__main__":
    r_dic = {}

    r_dic["model"] = "FNO"

    r_dic["train_path"] = "mul_tri_train.mat"
    r_dic["val_path"] = "mul_tri_val.mat"
    r_dic["test_path"] = "mul_tri_test.mat"

    r_dic["Data_path"] = DATA_PATH
    r_dic["train_len"] = 1000
    r_dic["val_len"] = 100
    r_dic["test_len"] = 100
    r_dic["resolution_datasets"] = 1023
    r_dic["batch_size"] = 8

    r_dic["boundary_condition"] = "dirichlet"
    r_dic["losstype"] = "H1"

    r_dic["xGN"] = False
    r_dic["subsample_nodes"] = 4
    r_dic["subsample_attn"] = 1
    r_dic["res_input"] = 256
    r_dic["res_output"] = 256

    r_dic["epochs"] = 300
    r_dic["feature_dim"] = 32
    r_dic["FNO_modes"] = 12

    r_dic["model_save_path"] = MODEL_PATH
    r_dic["model_name"] = "FNO_multiscale_res256.pt"
    r_dic["result_name"] = str(r_dic["model_name"][0:-3]) + ".pkl"
    r_dic["mat_name"] = str(r_dic["model_name"][0:-3]) + ".mat"

    print(r_dic)
    train_model(r_dic)

    print("END")
