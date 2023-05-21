from train import *

SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = 'E:/SJTU/torch_project/HT-net/data'
DATA_PATH = os.path.join(SRC_ROOT, 'data')
MODEL_PATH = os.path.join(SRC_ROOT, 'models')




if __name__ == "__main__":
    R_dic = {}

    R_dic['train_path'] = 'NavierStokes_V1e-5_N1200_T20.mat'
    R_dic['test_path'] = 'NavierStokes_V1e-5_N1200_T20.mat'
    R_dic['Data_path'] = DATA_PATH
    R_dic['boundary_condition'] = None
    R_dic['train_len'] = 1000
    R_dic['test_len'] = 200
    R_dic['resolution_datasets'] = 64  # resolution of data sets
    R_dic['batch_size'] = 8
    R_dic['T'] = 10
    R_dic['T_in'] = 10
    R_dic['in_dim'] = R_dic['T_in']

    # R_dic['train_path'] = 'ns_V1e-3_N5000_T50.mat'
    # R_dic['test_path'] = 'ns_V1e-3_N5000_T50.mat'
    # R_dic['boundary_condition'] = None
    # R_dic['train_len'] = 1000
    # R_dic['test_len'] = 100
    # R_dic['resolution_datasets'] = 64  # resolution of data sets
    # R_dic['batch_size'] = 8
    # R_dic['T'] = 40
    # R_dic['T_in'] = 10
    # R_dic['in_dim'] = R_dic['T_in']

    # R_dic['train_path'] = 'ns_V1e-4_N10000_T30.mat'
    # R_dic['test_path'] = 'ns_V1e-4_N10000_T30.mat'
    # R_dic['boundary_condition'] = None
    # R_dic['train_len'] = 1000
    # R_dic['test_len'] = 100
    # R_dic['resolution_datasets'] = 64  # resolution of data sets
    # R_dic['batch_size'] = 8
    # R_dic['T'] = 20
    # R_dic['T_in'] = 10
    # R_dic['in_dim'] = R_dic['T_in']

    # R_dic['train_path'] = 'ns_V1e-4_N10000_T30.mat'
    # R_dic['test_path'] = 'ns_V1e-4_N10000_T30.mat'
    # R_dic['boundary_condition'] = None
    # R_dic['train_len'] = 9000
    # R_dic['test_len'] = 1000
    # R_dic['resolution_datasets'] = 64  # resolution of data sets
    # R_dic['batch_size'] = 8
    # R_dic['T'] = 20
    # R_dic['T_in'] = 10
    # R_dic['in_dim'] = R_dic['T_in']

    R_dic['losstype'] = 'H1'

    R_dic['xGN'] = False
    R_dic['subsample_nodes'] = 1
    R_dic['subsample_attn'] = 1
    R_dic['patch_padding'] = 1
    R_dic['patch_size'] = 3
    R_dic['res_input'] = int((R_dic['resolution_datasets'] - 1) / R_dic['subsample_nodes'] + 1)
    R_dic['res_att'] = int((R_dic['res_input'] - R_dic['patch_size'] + 2 * R_dic['patch_padding']) / R_dic['subsample_attn'] + 1)
    R_dic['res_output'] = 64

    R_dic['epochs'] = 500
    R_dic['feature_dim'] = 20  # feature dim, in order to enhance expressiveness
    R_dic['window_size'] = [4, 4, 4]
    R_dic['depths'] = [2, 4, 2]
    R_dic['num_heads'] = [1, 1, 1]

    R_dic['F_modes'] = 12  # modes of FNO
    R_dic['F_width'] = 20
    R_dic['num_spectral_layers'] = 4
    R_dic['mlp_hidden_dim'] = 128
    R_dic['F_padding'] = 5


    '''
        save path
    '''
    R_dic['model_save_path'] = MODEL_PATH
    R_dic['model_name'] = 'NS_1e-5_N1200.pt'
    R_dic['result_name'] = str(R_dic['model_name'][0:-3]) + '.pkl'
    R_dic['mat_name'] = str(R_dic['model_name'][0:-3]) + '.mat'



    '''
        God blessed me！
    '''

    print(R_dic)
    train_NS_model(R_dic)

    print('END')


