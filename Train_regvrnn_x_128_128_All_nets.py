
### ===========================================================================
### Libraries =================================================================
import torch
import torch.utils
import torch.utils.data
from torch.autograd import Variable
# import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
# from torchvision.utils import save_image
import pickle
from time import time
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from utils import imgaussian
# from utils_vis_torch_hrt_frm import vis_torch_hrt_frm
from utils_vis_latent import vis_latent
from model_regvrnn_x_128_128_0_0 import VRNN_model

print(chr(12))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### ===========================================================================
### hyperparameters ===========================================================
frm_siz = 7
model_name = 'model_regvrnn_' + str(frm_siz) + '_128_128_0_0_abl_No_H'

N_epoch = 70
batch_siz = 10

z_dim = 64
sigma_img_blur = 0.5

lr_rate = 1e-3

if frm_siz == 7:
    prefix_data = r"F:\Dataset Source Main\Img_Mask_Dataset_7_128_128\Data_LAX_Grt_4620_7_128_128"

elif frm_siz == 14:
    prefix_data = r"F:\Dataset Source Main\Img_Mask_Dataset_14_128_128\Data_LAX_Grt_4620_14_128_128"

elif frm_siz == 25:
    prefix_data = r"F:\Dataset Source Main\Img_Mask_Dataset_25_128_128\Data_LAX_Grt_4620_25_128_128"

result_path = "./Weight_Results/"
try:
    os.mkdir(result_path)
except OSError:
    pass

### ===========================================================================
### Dataset load ==============================================================
N_train = 4000
N_val = 620

with open(prefix_data, "rb") as f:
    [img_all_frm_pack, _, _, img_pack_names] = pickle.load(f)

img_all_blur = np.zeros ([*img_all_frm_pack.shape])
for i in range(img_all_frm_pack.shape[0]):
    for j in range(img_all_frm_pack.shape[1]):
        img_all_blur[i,j] = imgaussian(img_all_frm_pack[i,j].astype(np.float64), sigma_img_blur)

train_mat = img_all_blur[:N_train]
train_mat /= 255.0
train_mat = torch.tensor(train_mat).unsqueeze(2).float()

# val_mat = img_all_blur[4000: 4000+N_val]
# val_mat /= 255.0
# val_mat = torch.tensor(val_mat).unsqueeze(2).float()

N_batch_train = int(np.ceil(N_train/batch_siz))
# N_batch_val = int(np.ceil(N_val/batch_siz))

train_names = img_pack_names[:N_train]
train_names = [int(name[:7]) for name,_,_ in train_names]

# val_names = img_pack_names[4000: 4000+N_val]
# val_names = [int(name[:7]) for name,_,_ in val_names]

### ===========================================================================
### shuffle data ==============================================================
def Fn_shuf_idx(N_data, N_frm, bch_siz):
    shuf_idx = np.arange(N_data)
    np.random.shuffle(shuf_idx)

    N_bch = int(np.ceil(N_data/bch_siz))
    idx_set = []

    for i in range(N_bch):
        idx_st = i * bch_siz
        idx_sp = np.min([(i+1)*bch_siz, N_data])

        sub_idx = shuf_idx[idx_st: idx_sp]
        idx_set.append(sub_idx)
    return idx_set

### ===========================================================================
### Create model ==============================================================
model = VRNN_model(device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= lr_step_size, gamma= lr_gamma, last_epoch=-1)
writer = SummaryWriter()

### ===========================================================================
### print model details =======================================================
print(model)
print("number of parameters: ", sum([param.numel() for param in model.parameters()]))

### ===========================================================================
### main ======================================================================
time0 = time()

for epoch in range(N_epoch):
    ### train =================================================================
    model.train()
    shuf_idx_train = Fn_shuf_idx(N_train, frm_siz, batch_siz)

    epoch_train_mse = 0
    epoch_train_grad = 0
    epoch_train_kld = 0
    epoch_train_klz = 0
    epoch_train_loss = 0
    epoch_train_mse_real = 0

    for batch_idx in range(N_batch_train):
        ls_1 = shuf_idx_train[batch_idx]
        x_in = Variable(train_mat[ls_1]).to(device)

        optimizer.zero_grad()
        x_out, displacement, mse_loss, grad_loss, kld_loss, klz_loss, loss, mse_loss_real = model(x_in)

        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        writer.add_scalar("train_mse_loss", mse_loss.item(), epoch*N_batch_train+batch_idx)
        writer.add_scalar("train_grad_loss", grad_loss.item(), epoch*N_batch_train+batch_idx)
        writer.add_scalar("train_klz_loss", klz_loss.item(), epoch*N_batch_train+batch_idx)
        writer.add_scalar("train_kld_loss", kld_loss.item(), epoch*N_batch_train+batch_idx)
        writer.add_scalar("train_loss", loss.item(), epoch*N_batch_train+batch_idx)
        writer.add_scalar("train_mse_loss_real", mse_loss_real.item(), epoch*N_batch_train+batch_idx)

        epoch_train_mse += mse_loss.item()
        epoch_train_grad += grad_loss.item()
        epoch_train_kld += kld_loss.item()
        epoch_train_klz += klz_loss.item()
        epoch_train_loss += loss.item()
        epoch_train_mse_real += mse_loss_real.item()

        if batch_idx % 5 == 0:
            print("Epoch: {} [{}/{} ({:.0f}%)]\t mse_loss: {:.5f} \t grad_loss: {:.5f} \t kld_loss: {:.5f} \t klz_loss: {:.5f} \t total_loss: {:.5f} \t mse_loss_real: {:.5f} \t".format(
                epoch, batch_idx, N_batch_train, batch_idx / N_batch_train*100.0,
                mse_loss.item(), grad_loss.item(), kld_loss.item(), klz_loss.item(), loss.item(), mse_loss_real.item()))

        ### visualize =========================================================
        # vis_torch_hrt_frm(x_in, x_out, displacement, b_idx = 0, cmap_typ= 'gray', jac_det_type= 2, name= '')

    epoch_train_mse /= N_batch_train
    epoch_train_grad /= N_batch_train
    epoch_train_kld /= N_batch_train
    epoch_train_klz /= N_batch_train
    epoch_train_loss /= N_batch_train
    epoch_train_mse_real /= N_batch_train

    writer.add_scalar("epoch_train_mse", epoch_train_mse, epoch)
    writer.add_scalar("epoch_train_grad", epoch_train_grad, epoch)
    writer.add_scalar("epoch_train_kld", epoch_train_kld, epoch)
    writer.add_scalar("epoch_train_klz", epoch_train_klz, epoch)
    writer.add_scalar("epoch_train_loss", epoch_train_loss, epoch)
    writer.add_scalar("epoch_train_mse_real", epoch_train_mse_real, epoch)

    ### eval latent space =====================================================
    if False:
        model.eval()
        z_mu_eval = np.empty((0, frm_siz, z_dim))
        z_var_eval = np.empty((0, frm_siz, z_dim))

        for batch_idx in range(N_batch_train):
            idx_start = batch_idx * batch_siz
            idx_stop = np.min([(batch_idx+1)*batch_siz, N_train])
            x_in = train_mat[idx_start: idx_stop].to(device)

            z_mu_temp, z_var_temp = model.latent_part(x_in)

            z_mu_eval = np.concatenate([z_mu_eval, z_mu_temp.detach().cpu().numpy()])
            z_var_eval = np.concatenate([z_var_eval, z_var_temp.detach().cpu().numpy()])

            if batch_idx > 100:
                break
        vis_latent(z_mu_eval[:1000])

Elapsed_Time = time()-time0
print('Elapsed Time (s): ', Elapsed_Time)

writer.close()

### ===========================================================================
### Save data =================================================================
torch.save(model.state_dict(), result_path + 'weights_' + model_name + '.pth')

print("==============")
print("Well Done!")
print("==============")