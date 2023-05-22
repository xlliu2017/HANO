import torch
import os
from models import *
from utils import *
from T_lossfunc import *
from torch.optim.lr_scheduler import OneCycleLR
from tqdm.auto import tqdm
import scipy.io as scio

def train_data(model, train_loader, loss_func, optimizer, lr_scheduler, train_len, losstype, device):
    model.train()
    loss_l2_epoch = 0.
    loss_h1_epoch = 0.

    for x, y in train_loader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        lossl2, lossh1 = loss_func(pred, y)
        if losstype == 'H1':
            lossh1.backward()
        elif losstype == 'L2':
            lossl2.backward()

        # nn.utils.clip_grad_norm_(model.parameters(), 0.99)  # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
        optimizer.step()

        loss_l2_epoch += lossl2
        loss_h1_epoch += lossh1

    loss_l2 = loss_l2_epoch / train_len
    loss_h1 = loss_h1_epoch / train_len
    lr = optimizer.param_groups[0]['lr']
    lr_scheduler.step()

    return loss_l2.item(), loss_h1.item(), lr

@torch.no_grad()
def test_data(model, test_loader, loss_func, test_len, device):
    model.eval()
    loss_l2_epoch = 0.
    loss_h1_epoch = 0.

    for x, y in test_loader:
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            lossl2, lossh1 = loss_func(pred, y)

            loss_l2_epoch += lossl2
            loss_h1_epoch += lossh1

    loss_l2 = loss_l2_epoch / test_len
    loss_h1 = loss_h1_epoch / test_len

    return loss_l2.item(), loss_h1.item()

def train_ns(model, train_loader, loss_func, optimizer, lr_scheduler, train_len, T_in, T, device):
    model.train()
    loss_l2_add = 0.
    loss_l2_full = 0.
    for xx, yy in train_loader:
        loss = 0
        batch_size = xx.shape[0]
        optimizer.zero_grad()
        xx = xx.to(device)
        yy = yy.to(device)
        for t in range(T):
            x = xx[:, :, :, t:t + T_in]
            x =  x.permute(0,3,1,2)
            y  = yy[..., t:t+1]
            im = model(x)
            loss += loss_func(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
        loss.backward()
        lossl2_full = loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))

        optimizer.step()
        loss_l2_add += loss
        loss_l2_full += lossl2_full

    loss_l2_add  = loss_l2_add / train_len
    loss_l2_full = loss_l2_full/ train_len

    lr = optimizer.param_groups[0]['lr']
    lr_scheduler.step()

    return loss_l2_add.item(), loss_l2_full.item(), lr

@torch.no_grad()
def test_ns(model, test_loader, loss_func, test_len, T, device):
    model.eval()
    loss_l2_add = 0.
    loss_l2_full = 0.
    for xx, yy in test_loader:
        loss = 0
        batch_size = xx.shape[0]
        xx = xx.to(device)
        yy = yy.to(device)
        for t in range(T):
            x = xx.permute(0, 3, 1, 2)

            y  = yy[..., t:t+1]
            im = model(x)
            loss += loss_func(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., 1:], im), dim=-1)

        lossl2_full = loss_func(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))

        loss_l2_add += loss
        loss_l2_full += lossl2_full

    loss_l2_add  = loss_l2_add / test_len
    loss_l2_full = loss_l2_full/ test_len

    return loss_l2_add.item(), loss_l2_full.item()

# =======================================================================================
def train_model(R_dic):
    # ==================== Load data ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_path = os.path.join(R_dic['Data_path'], R_dic['train_path'])
    val_path = os.path.join(R_dic['Data_path'], R_dic['val_path'])
    test_path = os.path.join(R_dic['Data_path'], R_dic['test_path'])

    x_train, y_train, x_normalizer, y_normalizer = \
        Data_load(train_path, R_dic['train_len'], res_input=R_dic['res_input'], res_output=R_dic['res_output'],
                  xGN=R_dic['xGN'], train_data=True)
    x_val, y_val, _, _ = \
        Data_load(val_path, R_dic['val_len'], res_input=R_dic['res_input'], res_output=R_dic['res_output'],
                  xGN=R_dic['xGN'], xnormalizer=x_normalizer, train_data=False)
    x_test, y_test, _, _ = \
        Data_load(test_path, R_dic['test_len'], res_input=R_dic['res_input'], res_output=R_dic['res_output'],
                  xGN=R_dic['xGN'], xnormalizer=x_normalizer, train_data=False)

    R_dic['y_norm'] = y_normalizer

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train.contiguous(), y_train.contiguous()),
        batch_size=R_dic['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_val.contiguous(), y_val.contiguous()),
        batch_size=R_dic['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test.contiguous(), y_test.contiguous()),
        batch_size=R_dic['batch_size'], shuffle=False)

    # ==================== Load model, optimizer, learning rate scheduler, loss function ====================
    if R_dic['model'] == 'HANO':
        model = HANO2d(R_dic)
    elif R_dic['model'] == 'FNO':
        model = FNO2d(R_dic)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-4)

    lr_scheduler = OneCycleLR(optimizer, max_lr=8e-4,
                              div_factor=4,
                              final_div_factor=8,
                              pct_start=0.2,
                              steps_per_epoch=1,
                              epochs=R_dic['epochs'])

    loss_func = H1Loss(res=R_dic['res_output'])
    loss_func.cuda(device)
    epochs = R_dic['epochs']

    loss_train = []
    loss_val = []
    loss_test = []
    lr_history = []
    stop_counter = 0
    best_val_metric = np.inf
    test_by_val = np.inf
    best_val_epoch = None

    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            train_l2, train_h1, lr = train_data(model, train_loader, loss_func, optimizer, lr_scheduler,
                                                train_len=R_dic['train_len'], losstype=R_dic['losstype'], device=device)
            loss_train.append([train_l2, train_h1])
            lr_history.append(lr)
            val_l2, val_h1 = test_data(model, val_loader, loss_func, test_len=R_dic['val_len'],
                                       device=device)
            loss_val.append([val_l2, val_h1])
            test_l2, test_h1 = test_data(model, test_loader, loss_func, test_len=R_dic['test_len'],
                                         device=device)
            loss_test.append([test_l2, test_h1])

            if val_l2 < best_val_metric:
                best_val_epoch = epoch
                best_val_metric = val_l2
                test_by_val = test_l2
                stop_counter = 0

                torch.save(model, os.path.join(R_dic['model_save_path'], R_dic['model_name']))

            else:
                stop_counter += 1

            desc = color(f"| Test L2 loss: {test_l2:.3e} ", color=Colors.blue)
            desc += color(f"| Test H1 loss: {test_h1:.3e} ", color=Colors.blue)
            desc += color(f"| test by val: {test_by_val:.3e} at epoch {best_val_epoch + 1}", color=Colors.green)
            desc += color(f" | early stop: {stop_counter} ", color=Colors.green)
            desc += color(f" | current lr: {lr:.3e}", color=Colors.magenta)

            desc_ep = color("", color=Colors.red)

            desc_ep += color(f"| Train L2 loss : {train_l2:.3e} ", color=Colors.red)
            desc_ep += color(f"| Train H1 loss : {train_h1:.3e} ", color=Colors.red)
            desc_ep += color(f"| Val L2 loss : {val_l2:.3e} ", color=Colors.yellow)

            desc_ep += desc
            pbar.set_description(desc_ep)
            pbar.update()

            result = dict(
                best_val_epoch=best_val_epoch,
                best_val_metric=best_val_metric,
                test_by_val=test_by_val,
                loss_train=np.asarray(loss_train),
                loss_test=np.asarray(loss_test),
                lr_history=np.asarray(lr_history),
                optimizer_state=optimizer.state_dict()
            )
            # save the result
            save_pickle(result, os.path.join(R_dic['model_save_path'], R_dic['result_name']))


    print('Model train END')
    print('Test L2 error' + str(test_by_val))

    s1 = R_dic['res_output']
    s2 = R_dic['res_input']
    z_true, z = torch.zeros(1, s1, s1).to(device), torch.zeros(1, s1, s1).to(device)
    xinput = torch.zeros(1, s2, s2).to(device)

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x).reshape(-1, s1, s1)
            y = y.reshape(-1, s1, s1)

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
    SAVE_PATH = os.path.join('results', R_dic['mat_name'])
    scio.savemat(SAVE_PATH, {'input': xinput, 'truth': z_true, 'output': z})


    print('Path of the result(.mat)' + str(SAVE_PATH))
    return result

def train_NS_model(R_dic):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ==================== Load data ====================
    train_path = os.path.join(R_dic['Data_path'], R_dic['train_path'])
    test_path = os.path.join(R_dic['Data_path'], R_dic['test_path'])

    x_train, y_train = \
        Data_NS(train_path, R_dic['train_len'], R_dic['T_in'], R_dic['T'], train_data=True)

    x_test, y_test = \
        Data_NS(test_path, R_dic['test_len'], R_dic['T_in'], R_dic['T'], train_data=False)


    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train.contiguous(), y_train.contiguous()),
        batch_size=R_dic['batch_size'], shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test.contiguous(), y_test.contiguous()),
        batch_size=R_dic['batch_size'], shuffle=False)

    torch.cuda.empty_cache()

    # ==================== Load model, optimizer, learning rate scheduler, loss function ====================
    model = HANO2d(R_dic)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-4)

    lr_scheduler = OneCycleLR(optimizer, max_lr=8e-4,
                              div_factor=4,
                              final_div_factor=8,
                              pct_start=0.2,
                              steps_per_epoch=1,
                              epochs=R_dic['epochs'])

    loss_func = LpLoss(size_average=False)
    epochs = R_dic['epochs']

    loss_train = []
    loss_test = []
    lr_history = []
    stop_counter = 0
    best_val_metric = np.inf
    best_val_epoch = None

    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            train_l2_add, train_l2_full, lr= train_ns(model, train_loader, loss_func, optimizer, lr_scheduler,
                                                   R_dic['train_len'], R_dic['T_in'], R_dic['T'], device=device)
            loss_train.append([train_l2_add, train_l2_full])
            lr_history.append(lr)

            test_l2_add, test_l2_full = test_ns(model, test_loader, loss_func, R_dic['test_len'], R_dic['T'], device=device)
            loss_test.append([test_l2_add, test_l2_full])

            if test_l2_full < best_val_metric:
                best_val_epoch = epoch
                best_val_metric = test_l2_full
                stop_counter = 0

            else:
                stop_counter += 1

            desc = color(f"| Test L2 add: {test_l2_add:.3e} ", color=Colors.blue)
            desc += color(f"| Test L2 full: {test_l2_full:.3e} ", color=Colors.blue)
            desc += color(f"| best val: {best_val_metric:.3e} at epoch {best_val_epoch + 1}", color=Colors.green)
            desc += color(f" | early stop: {stop_counter} ", color=Colors.green)
            desc += color(f" | current lr: {lr:.3e}", color=Colors.magenta)

            desc_ep = color("", color=Colors.red)

            desc_ep += color(f"| Train L2 add: {train_l2_add:.3e} ", color=Colors.red)
            desc_ep += color(f"| Train L2 full: {train_l2_full:.3e} ", color=Colors.red)

            desc_ep += desc
            pbar.set_description(desc_ep)
            pbar.update()

            result = dict(
                best_val_epoch=best_val_epoch,
                best_val_metric=best_val_metric,
                loss_train=np.asarray(loss_train),
                loss_test=np.asarray(loss_test),
                lr_history=np.asarray(lr_history),
                optimizer_state=optimizer.state_dict()
            )
            save_pickle(result, os.path.join(R_dic['model_save_path'], R_dic['result_name']))

    torch.save(model, os.path.join(R_dic['model_save_path'], R_dic['model_name']))

    print('Relative L2 error:' + str(test_l2_full))

