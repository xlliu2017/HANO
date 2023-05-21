from train import *

SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SRC_ROOT, 'data')
MODEL_PATH = os.path.join(SRC_ROOT, 'models')


if __name__ == "__main__":
    R_dic = {}

    R_dic['model'] = 'HANO'

    R_dic['train_path'] = 'darcy_rough_train.mat'
    R_dic['val_path'] = 'darcy_rough_val.mat'
    R_dic['test_path'] = 'darcy_rough_test.mat'
    R_dic['Data_path'] = DATA_PATH
    R_dic['train_len'] = 1280
    R_dic['val_len'] = 112
    R_dic['test_len'] = 112
    R_dic['resolution_datasets'] = 512  # resolution of data sets
    R_dic['batch_size'] = 8

    R_dic['boundary_condition'] = 'dirichlet'
    R_dic['losstype'] = 'H1'

    R_dic['xGN'] = True
    R_dic['subsample_nodes'] = 1
    R_dic['subsample_attn'] = 2
    R_dic['patch_padding'] = 1
    R_dic['patch_size'] = 4
    R_dic['res_input'] = int((R_dic['resolution_datasets'] - 1) / R_dic['subsample_nodes'] + 1)
    R_dic['res_att'] = int((R_dic['res_input'] - R_dic['patch_size'] + 2 * R_dic['patch_padding']) / R_dic['subsample_attn'] + 1)
    R_dic['res_output'] = 256

    R_dic['epochs'] = 500
    R_dic['feature_dim'] = 64  # feature dim, in order to enhance expressiveness
    R_dic['window_size'] = [4, 4, 4]
    R_dic['depths'] = [1, 1, 1]
    R_dic['num_heads'] = [1, 1, 1]

    R_dic['F_modes'] = 12
    R_dic['F_width'] = 64
    R_dic['num_spectral_layers'] = 5
    R_dic['mlp_hidden_dim'] = 128
    R_dic['F_padding'] = 5
    '''
        save path
    '''
    R_dic['model_save_path'] = MODEL_PATH
    R_dic['model_name'] = 'darcyrough_res256.pt'
    R_dic['result_name'] = str(R_dic['model_name'][0:-3]) + '.pkl'
    R_dic['mat_name'] = str(R_dic['model_name'][0:-3]) + '.mat'

    '''
     God blessed me！
    '''

    print(R_dic)
    train_model(R_dic)

    print('END')

