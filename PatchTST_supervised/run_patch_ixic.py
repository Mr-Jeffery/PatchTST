from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
# from layers.PatchTST_backbone import PatchTST_backbone
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import argparse
import random
import os
import time

import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='costom', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='^IXIC.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

configs = parser.parse_configs()

# initialize model parameters
c_in:int
context_window:int
target_window:int
patch_len:int
stride:int
max_seq_len:[int]=1024 #Optional
n_layers:int=3
d_model=128
n_heads=16
d_k:[int]=None #Optional
d_v:[int]=None #Optional
d_ff:int=256
norm:str='BatchNorm'
attn_dropout:float=0.
dropout:float=0.
act:str="gelu"
key_padding_mask:bool='auto',
padding_var:[int]=None #Optional
attn_mask:[nn.Tensor]=None #Optional
res_attention:bool=True
pre_norm:bool=False
store_attn:bool=False
pe:str='zeros'
learn_pe:bool=True
fc_dropout:float=0.
head_dropout = 0
padding_patch = None,
pretrain_head:bool=False
head_type = 'flatten'
individual = False
revin = True
affine = True
subtract_last = False
verbose:bool=False

# load model parameters
c_in = configs.enc_in
context_window = configs.seq_len
target_window = configs.pred_len

n_layers = configs.e_layers
n_heads = configs.n_heads
d_model = configs.d_model
d_ff = configs.d_ff
dropout = configs.dropout
fc_dropout = configs.fc_dropout
head_dropout = configs.head_dropout

individual = configs.individual

patch_len = configs.patch_len
stride = configs.stride
padding_patch = configs.padding_patch

revin = configs.revin
affine = configs.affine
subtract_last = configs.subtract_last

decomposition = configs.decomposition
kernel_size = configs.kernel_size

# random seed
fix_seed = configs.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)




# model
model = PatchTST(c_in=c_in, context_window = context_window, 
    target_window=target_window, patch_len=patch_len, stride=stride, 
    max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
    n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
    dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
    attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
    pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
    pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
    subtract_last=subtract_last, verbose=verbose).float()


# load GPU parameters
use_gpu = True if torch.cuda.is_available() and configs.use_gpu else False
if use_gpu and configs.use_multi_gpu:
    configs.dvices = configs.devices.replace(' ', '')
    device_ids = configs.devices.split(',')
    device_ids = [int(id_) for id_ in device_ids]
    model = nn.DataParallel(model, device_ids=device_ids)

if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        configs.gpu) if not configs.use_multi_gpu else configs.devices
    device = torch.device('cuda:{}'.format(configs.gpu))
    print('Use GPU: cuda:{}'.format(configs.gpu))
else:
    device = torch.device('cpu')
    print('Use CPU')

model.to(device)

setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
    configs.model_id,
    configs.model,
    configs.data,
    configs.features,
    configs.seq_len,
    configs.label_len,
    configs.pred_len,
    configs.d_model,
    configs.n_heads,
    configs.e_layers,
    configs.d_layers,
    configs.d_ff,
    configs.factor,
    configs.embed,
    configs.distil,
    configs.des,0)


train_data, train_loader = data_provider(configs,flag='train')
vali_data, vali_loader = data_provider(configs,flag='val')
test_data, test_loader = data_provider(configs,flag='test')

path = os.path.join(configs.checkpoints, setting)
if not os.path.exists(path):
    os.makedirs(path)

time_now = time.time()

train_steps = len(train_loader)
early_stopping = EarlyStopping(patience=configs.patience, verbose=True)

model_optim = optim.Adam(model.parameters(), lr=configs.learning_rate)
criterion = nn.MSELoss()

if configs.use_amp:
    scaler = torch.cuda.amp.GradScaler()
    
scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                    steps_per_epoch = train_steps,
                                    pct_start = configs.pct_start,
                                    epochs = configs.train_epochs,
                                    max_lr = configs.learning_rate)


def vali(model, device, vali_data, vali_loader, criterio,args):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if 'Linear' in args.model or 'TST' in args.model:
                        outputs = model(batch_x)
                    else:
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if 'Linear' in args.model or 'TST' in args.model:
                    outputs = model(batch_x)
                else:
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            loss = criterion(pred, true)

            total_loss.append(loss)
    total_loss = np.average(total_loss)
    model.train()
    return total_loss

for epoch in range(configs.train_epochs):
    iter_count = 0
    train_loss = []

    model.train()
    epoch_time = time.time()
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        iter_count += 1
        model_optim.zero_grad()
        batch_x = batch_x.float().to(device)

        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -configs.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :configs.label_len, :], dec_inp], dim=1).float().to(device)

        # encoder - decoder
        if configs.use_amp:
            with torch.cuda.amp.autocast():
                if 'Linear' in configs.model or 'TST' in configs.model:
                    outputs = model(batch_x)
                else:
                    if configs.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if configs.features == 'MS' else 0
                outputs = outputs[:, -configs.pred_len:, f_dim:]
                batch_y = batch_y[:, -configs.pred_len:, f_dim:].to(device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
        else:
            if 'Linear' in configs.model or 'TST' in configs.model:
                    outputs = model(batch_x)
            else:
                if configs.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
            # print(outputs.shape,batch_y.shape)
            f_dim = -1 if configs.features == 'MS' else 0
            outputs = outputs[:, -configs.pred_len:, f_dim:]
            batch_y = batch_y[:, -configs.pred_len:, f_dim:].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

        if (i + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((configs.train_epochs - epoch) * train_steps - i)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = time.time()

        if configs.use_amp:
            scaler.scale(loss).backward()
            scaler.step(model_optim)
            scaler.update()
        else:
            loss.backward()
            model_optim.step()
            
        if configs.lradj == 'TST':
            adjust_learning_rate(model_optim, scheduler, epoch + 1, configs, printout=False)
            scheduler.step()

    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    train_loss = np.average(train_loss)
    vali_loss = vali(model=model, device=device, vali_data=vali_data, vali_loader=vali_loader, criterion=criterion, args=configs)
    test_loss = vali(model=model, device=device, vali_data=test_data, vali_loader=test_loader, criterion=criterion, args=configs)

    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
        epoch + 1, train_steps, train_loss, vali_loss, test_loss))
    early_stopping(vali_loss, model, path)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    if configs.lradj != 'TST':
        adjust_learning_rate(model_optim, scheduler, epoch + 1, configs)
    else:
        print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

best_model_path = path + '/' + 'checkpoint.pth'
model.load_state_dict(torch.load(best_model_path))


def test(self, setting, test=0):
    test_data, test_loader = self._get_data(flag='test')
    
    if test:
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    preds = []
    trues = []
    inputx = []
    folder_path = './test_results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    self.model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -configs.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :configs.label_len, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if configs.use_amp:
                with torch.cuda.amp.autocast():
                    if 'Linear' in configs.model or 'TST' in configs.model:
                        outputs = self.model(batch_x)
                    else:
                        if configs.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if 'Linear' in configs.model or 'TST' in configs.model:
                        outputs = self.model(batch_x)
                else:
                    if configs.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if configs.features == 'MS' else 0
            # print(outputs.shape,batch_y.shape)
            outputs = outputs[:, -configs.pred_len:, f_dim:]
            batch_y = batch_y[:, -configs.pred_len:, f_dim:].to(self.device)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

            preds.append(pred)
            trues.append(true)
            inputx.append(batch_x.detach().cpu().numpy())
            if i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    if configs.test_flop:
        test_params_flop((batch_x.shape[1],batch_x.shape[2]))
        exit()
    preds = np.array(preds)
    trues = np.array(trues)
    inputx = np.array(inputx)

    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

    # result save
    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
    f = open("result.txt", 'a')
    f.write(setting + "  \n")
    f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
    f.write('\n')
    f.write('\n')
    f.close()

    # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
    np.save(folder_path + 'pred.npy', preds)
    # np.save(folder_path + 'true.npy', trues)
    # np.save(folder_path + 'x.npy', inputx)
    return

def predict(self, setting, load=False):
    pred_data, pred_loader = self._get_data(flag='pred')

    if load:
        path = os.path.join(configs.checkpoints, setting)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

    preds = []

    self.model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros([batch_y.shape[0], configs.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
            dec_inp = torch.cat([batch_y[:, :configs.label_len, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if configs.use_amp:
                with torch.cuda.amp.autocast():
                    if 'Linear' in configs.model or 'TST' in configs.model:
                        outputs = self.model(batch_x)
                    else:
                        if configs.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if 'Linear' in configs.model or 'TST' in configs.model:
                    outputs = self.model(batch_x)
                else:
                    if configs.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            pred = outputs.detach().cpu().numpy()  # .squeeze()
            preds.append(pred)

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    # result save
    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(folder_path + 'real_prediction.npy', preds)

    return
