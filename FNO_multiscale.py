from train import *

SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = os.path.join(SRC_ROOT, 'data')
DATA_PATH = 'E:/SJTU/torch_project/HT-net/data'
MODEL_PATH = os.path.join(SRC_ROOT, 'models')


if __name__ == "__main__":
    R_dic = {}

    R_dic['model'] = 'FNO'

    # R_dic['train_path'] = 'mul_tri_train.mat'
    # R_dic['val_path'] = 'mul_tri_val.mat'
    # R_dic['test_path'] = 'mul_tri_test.mat'
    R_dic['train_path'] = 'gamblet_train.mat'
    R_dic['val_path'] = 'gamblet_val.mat'
    R_dic['test_path'] = 'gamblet_test.mat'

    R_dic['Data_path'] = DATA_PATH
    R_dic['train_len'] = 1000
    R_dic['val_len'] = 100
    R_dic['test_len'] = 100
    R_dic['resolution_datasets'] = 1023  # resolution of data sets
    R_dic['batch_size'] = 8

    R_dic['boundary_condition'] = 'dirichlet'
    R_dic['losstype'] = 'H1'

    R_dic['xGN'] = False
    R_dic['subsample_nodes'] = 4
    R_dic['subsample_attn'] = 1
    R_dic['res_input'] = 256
    R_dic['res_output'] = 256

    R_dic['epochs'] = 300
    R_dic['feature_dim'] = 32  # feature dim, in order to enhance expressiveness
    R_dic['FNO_modes'] = 12
    '''
        save path
    '''
    R_dic['model_save_path'] = MODEL_PATH
    R_dic['model_name'] = 'multiscale_res256.pt'
    R_dic['result_name'] = str(R_dic['model_name'][0:-3]) + '.pkl'
    R_dic['mat_name'] = str(R_dic['model_name'][0:-3]) + '.mat'
    R_dic['fig_name'] = str(R_dic['model_name'][0:-3]) + '.png'

    '''
        save path
    '''
    R_dic['model_save_path'] = MODEL_PATH
    R_dic['model_name'] = 'FNO_multiscale_res256.pt'
    R_dic['result_name'] = str(R_dic['model_name'][0:-3]) + '.pkl'
    R_dic['mat_name'] = str(R_dic['model_name'][0:-3]) + '.mat'

    '''
     God blessed me！
    '''

    print(R_dic)
    train_model(R_dic)

    print('END')

