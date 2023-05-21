import torch
from train import *

SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = 'E:/SJTU/torch_project/HT-net/data'
MODEL_PATH = os.path.join(SRC_ROOT, 'models')

def test_model(R_dic):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(MODEL_PATH, R_dic['modelname'])
    model = torch.load(model_path)
    model = model.to(device)

    train_path = os.path.join(DATA_PATH, R_dic['train_path'])
    test_path = os.path.join(DATA_PATH, R_dic['test_path'])

    x_train, y_train, x_normalizer, y_normalizer = \
        Data_load(train_path, R_dic['train_len'], res_input=R_dic['res_input'], res_output=R_dic['res_output'],
                  xGN=R_dic['xGN'], train_data=True)
    x_test, y_test, _, _ = \
        Data_load(test_path, R_dic['test_len'], res_input=R_dic['res_input'], res_output=R_dic['res_output'],
                  xGN=R_dic['xGN'], xnormalizer=x_normalizer, train_data=False)

    R_dic['y_norm'] = y_normalizer

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test.contiguous(), y_test.contiguous()),
        batch_size=R_dic['batch_size'], shuffle=False)

    # ------------------------------compute predict solution----------------------------------------------
    model.eval()
    s1 = R_dic['res_output']
    s2 = R_dic['res_input']
    z_true, z = torch.zeros(1, s1, s1).to(device), torch.zeros(1, s1, s1).to(device)
    xinput = torch.zeros(1, s2, s2).to(device)

    loss_func = H1Loss(res=R_dic['res_output'])
    l2error = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x).reshape(-1, s1, s1)
            y = y.reshape(-1, s1, s1)
            lossl2, _= loss_func(out, y)
            l2error += lossl2

            if R_dic['xGN']:
                x_normalizer = x_normalizer.to(device)
                x = x_normalizer.decode(x.reshape(-1, s2, s2))
            else:
                x = x.reshape(-1, s2, s2)
            xinput = torch.cat((xinput, x), dim=0)
            z_true = torch.cat((z_true, y), dim=0)
            z = torch.cat((z, out), dim=0)

    xinput = xinput[1:, ...].cpu().numpy()
    z_true = z_true[1:, ...].cpu().numpy()
    z = z[1:, ...].cpu().numpy()
    scio.savemat(R_dic['savemat_name'], {'input': xinput, 'truth': z_true, 'output': z})

    l2error = l2error / R_dic['test_len']
    print(f'l2error = {l2error}')

if __name__ == "__main__":
    # ==================================================darcy rough====================================================================
    R_dic = {}
    R_dic['modelname'] = 'darcyrough_res256.pt'
    R_dic['savemat_name'] = 'results/darcyrough_res256.mat'
    R_dic['train_path'] = 'darcy_rough_train.mat'
    R_dic['test_path'] = 'darcy_rough_test.mat'

    R_dic['Data_path'] = DATA_PATH
    R_dic['train_len'] = 1280
    R_dic['val_len'] = 112
    R_dic['test_len'] = 112
    R_dic['resolution_datasets'] = 512  # resolution of data sets
    R_dic['batch_size'] = 8

    R_dic['xGN'] = True
    R_dic['subsample_nodes'] = 1
    R_dic['subsample_attn'] = 2
    R_dic['patch_padding'] = 1
    R_dic['patch_size'] = 4
    R_dic['res_input'] = int((R_dic['resolution_datasets'] - 1) / R_dic['subsample_nodes'] + 1)
    R_dic['res_att'] = int(
        (R_dic['res_input'] - R_dic['patch_size'] + 2 * R_dic['patch_padding']) / R_dic['subsample_attn'] + 1)
    R_dic['res_output'] = 256

    test_model(R_dic)
    print('END')

    # ==================================================multiscale====================================================================
    # R_dic = {}
    # R_dic['modelname'] = 'multiscale_res256.pt'
    # R_dic['savemat_name'] = 'results/multiscale_res256.mat'
    # R_dic['train_path'] = 'mul_tri_train.mat'
    # R_dic['test_path'] = 'mul_tri_test.mat'
    # R_dic['Data_path'] = DATA_PATH
    # R_dic['train_len'] = 1000
    # R_dic['val_len'] = 100
    # R_dic['test_len'] = 100
    # R_dic['resolution_datasets'] = 1023  # resolution of data sets
    # R_dic['batch_size'] = 8
    #
    # R_dic['xGN'] = False
    # R_dic['subsample_nodes'] = 1
    # R_dic['subsample_attn'] = 4
    # R_dic['patch_padding'] = 2
    # R_dic['patch_size'] = 4
    # R_dic['res_input'] = int((R_dic['resolution_datasets'] - 1) / R_dic['subsample_nodes'] + 1)
    # R_dic['res_att'] = int((R_dic['res_input'] - R_dic['patch_size'] + 2 * R_dic['patch_padding']) / R_dic['subsample_attn'] + 1)
    # R_dic['res_output'] = 256
    #
    # test_model(R_dic)
    # print('END')