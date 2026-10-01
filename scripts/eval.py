import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hano.trainer import test_model

SRC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_PATH = os.path.join(SRC_ROOT, "data")
MODEL_PATH = os.path.join(SRC_ROOT, "models")


if __name__ == "__main__":
    r_dic = {}
    r_dic["modelname"] = "darcyrough_res256.pt"
    r_dic["savemat_name"] = os.path.join(SRC_ROOT, "results", "darcyrough_res256.mat")
    r_dic["train_path"] = "darcy_rough_train.mat"
    r_dic["test_path"] = "darcy_rough_test.mat"

    r_dic["Data_path"] = DATA_PATH
    r_dic["Model_path"] = MODEL_PATH
    r_dic["train_len"] = 1280
    r_dic["val_len"] = 112
    r_dic["test_len"] = 112
    r_dic["resolution_datasets"] = 512
    r_dic["batch_size"] = 8

    r_dic["xGN"] = True
    r_dic["subsample_nodes"] = 1
    r_dic["subsample_attn"] = 2
    r_dic["patch_padding"] = 1
    r_dic["patch_size"] = 4
    r_dic["res_input"] = int((r_dic["resolution_datasets"] - 1) / r_dic["subsample_nodes"] + 1)
    r_dic["res_att"] = int((r_dic["res_input"] - r_dic["patch_size"] + 2 * r_dic["patch_padding"]) / r_dic["subsample_attn"] + 1)
    r_dic["res_output"] = 256

    test_model(r_dic)
    print("END")
