
import torch
import torch.nn as nn
# import numpy as np

from utils.convlstm_layer import ConvLSTM
from utils.stn_2d_layer import stn_2d
from utils.loss_function import loss_fn

### network layers ============================================================
def conv_down(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1))

def conv_up(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))

### dragnet model =============================================================
class dragnet(nn.Module):
    def __init__(self, device):
        super(dragnet, self).__init__()

        ### coefficients of loss terms
        self.coef_klz = 2e-4
        self.coef_smooth = 0.03
        self.coef_kld = 1e-4

        self.z_dim = 64       # latent space dimension
        self.device = device
        self.LRelu = nn.LeakyReLU(0.2)

        ### z_prior -----------------------------------------------------------
        self.conv1_z_prior = conv_down(16, 4)
        self.fc_mu_z_prior = nn.Linear(4*16*16, self.z_dim)
        self.fc_logvar_z_prior = nn.Linear(4*16*16, self.z_dim)

        ### phi_x -------------------------------------------------------------
        self.conv1_phi_x = conv_down(1, 32)
        self.conv2_phi_x = conv_down(32, 32)

        ### net_infer ---------------------------------------------------------
        self.conv1_infer = conv_down(32+16, 16)
        self.fc_mu_infer = nn.Linear(16*16*16, self.z_dim)
        self.fc_logvar_infer = nn.Linear(16*16*16, self.z_dim)

        ### phi_z -------------------------------------------------------------
        self.fc1_phi_z = nn.Linear(self.z_dim, 16*16*16)
        self.conv1_phi_z = conv_up(16, 32)

        ### net_gen -----------------------------------------------------------
        self.conv1_gen = conv_up(32+32, 32)
        self.conv2_gen = conv_up(32, 32)
        self.conv_mu_gen = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.conv_logvar_gen = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.conv_v_gen = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)

        ### ConvLSTM layer ----------------------------------------------------
        self.ConvLSTM = ConvLSTM(input_size=(32, 32),
                                 input_dim= 32+32,
                                 hidden_dim=[32, 16],
                                 kernel_size=[3,3],
                                 num_layers=2,
                                 device= device,
                                 batch_first=True,
                                 bias=True)

    ### sub_nets ==============================================================
    def net_z_prior(self, x):
        x = self.LRelu(self.conv1_z_prior(x))
        x = x.view(x.size(0), -1)
        z_prior_mu = self.fc_mu_z_prior(x)
        z_prior_logvar = self.fc_logvar_z_prior(x)
        return z_prior_mu, z_prior_logvar

    def net_phi_x(self, x):
        x =self.LRelu(self.conv1_phi_x(x))
        x = self.LRelu(self.conv2_phi_x(x))
        return x

    def net_infer(self, x):
        x = self.LRelu(self.conv1_infer(x))
        x = x.view(x.size(0), -1)
        z_mu = self.fc_mu_infer(x)
        z_logvar = self.fc_logvar_infer(x)
        return z_mu, z_logvar

    def zt_reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def covariance_d(self, log_var, log_v):
        v_m = torch.exp(log_v).permute(0,2,3,1).unsqueeze(-1)
        cov_m = torch.matmul(v_m, v_m.permute(0,1,2,4,3))
        temp1 = torch.exp(log_var)[:,0] + 1e-6
        cov_m[:,:,:,0,0] = cov_m[:,:,:,0,0] + temp1
        cov_m[:,:,:,1,1] = cov_m[:,:,:,1,1] + temp1
        return cov_m

    def dt_reparameterize(self, mu, cov):
        mu_m = mu.permute(0,2,3,1).unsqueeze(-1)
        eps_m = torch.randn_like(mu_m)
        cov_m = 0.5 * cov
        x = mu_m + torch.matmul(cov_m, eps_m)
        x = x.squeeze(-1)
        x = x.permute(0,3,1,2)
        return x

    def net_phi_z(self, x):
        x = self.fc1_phi_z(x)
        x = x.view(-1, 16, 16, 16)
        x = self.LRelu(self.conv1_phi_z(x))
        return x

    def net_gen(self, x):
        x = self.LRelu(self.conv1_gen(x))
        x = self.LRelu(self.conv2_gen(x))
        d_mu = self.conv_mu_gen(x)
        d_logvar = self.conv_logvar_gen(x)
        d_log_v = self.conv_v_gen(x)
        return d_mu, d_logvar, d_log_v

    def net_recurrent(self, x, hidden_state):
        h_recurrent, hidden_state= self.ConvLSTM(x, hidden_state)
        return h_recurrent, hidden_state

    ### forward (model structure) =============================================
    def forward(self, seq_in):
        batch_size, frame_size, _, h_img, w_img = seq_in.shape

        ### Initial value of loss terms
        mse_loss = 0.0
        klz_loss = 0.0
        smooth_loss = 0.0
        kld_loss = 0.0

        ### Initialize the recurrent layer with zeros
        h_recurrent = torch.zeros([batch_size, 16, 32, 32], device=self.device)   # Recurrent layer state

        ### Initialize the hidden state of the ConvLSTM module
        hidden_state = self.ConvLSTM._init_hidden(batch_size=batch_size)   # Hidden state initialization

        ### Initialize empty tensors to store the output displacement and sequence
        dis_out = torch.empty([batch_size, frame_size, 2, w_img, w_img], device=self.device)   # Output displacement tensor
        seq_out = torch.empty([batch_size, frame_size, 1, w_img, w_img], device=self.device)   # Output sequence tensor

        for t in range(frame_size+1):
            idx_cur = t % frame_size           ### current index
            idx_pas = (t-1) % frame_size       ### past index

            x_cur = seq_in[:, idx_cur]         ### current image
            x_pas = seq_in[:, idx_pas]         ### past image

            ### z_prior -------------------------------------------------------
            z_prior_mu, z_prior_logvar = self.net_z_prior(h_recurrent)

            ### phi_x ---------------------------------------------------------
            phi_x = self.net_phi_x(x_cur)

            ### net_infer -----------------------------------------------------
            phi_x_h = torch.cat([phi_x, h_recurrent], 1)
            z_mu, z_logvar = self.net_infer(phi_x_h)

            ### z_t reparameterize --------------------------------------------
            z_t = self.zt_reparameterize(z_mu, z_logvar)

            ### phi_z ---------------------------------------------------------
            phi_z = self.net_phi_z(z_t)

            ### net_gen -------------------------------------------------------
            phi_xp = self.net_phi_x(x_pas)
            phi_z_h = torch.cat([phi_z, phi_xp], 1)
            d_mu, d_logvar, d_log_v = self.net_gen(phi_z_h)

            ### d_t reparameterize --------------------------------------------
            d_cov = self.covariance_d(d_logvar, d_log_v)
            d_t = self.dt_reparameterize(d_mu, d_cov)

            ### x_out ---------------------------------------------------------
            x_out = stn_2d(d_t[:, [1,0]], x_pas, self.device)

            ### net_recurrent -------------------------------------------------
            phi_z_x = torch.cat([phi_z, phi_x], 1)
            h_recurrent, hidden_state= self.net_recurrent(phi_z_x, hidden_state)

            if t > 0:
                ### store results ---------------------------------------------
                seq_out[:, idx_cur] = x_out
                dis_out[:, idx_cur] = d_t

                ### compute loss ----------------------------------------------
                mse_loss_t, klz_loss_t, smooth_loss_t, kld_loss_t = loss_fn(x_cur, x_out,
                                                                            z_mu, z_logvar, z_prior_mu, z_prior_logvar,
                                                                            d_t, d_mu, d_cov)

                mse_loss += mse_loss_t
                klz_loss += klz_loss_t
                smooth_loss += smooth_loss_t
                kld_loss += kld_loss_t

        klz_loss *= self.coef_klz
        smooth_loss *= self.coef_smooth
        kld_loss *= self.coef_kld

        total_loss = mse_loss + klz_loss + smooth_loss + kld_loss
        return seq_out, dis_out, mse_loss, klz_loss, smooth_loss, kld_loss, total_loss

    ### generation from a single frame ========================================
    def gen_one_frame(self, x0, frame_size):
        batch_size, _, h_img, w_img = x0.shape

        ### Initialize the recurrent layer with zeros
        h_recurrent = torch.zeros([batch_size, 16, 32, 32], device=self.device)   # Recurrent layer state

        ### Initialize the hidden state of the ConvLSTM module
        hidden_state = self.ConvLSTM._init_hidden(batch_size=batch_size)   # Hidden state initialization

        ### Initialize empty tensors to store the Generated displacement and sequence
        dis_gen = torch.empty([batch_size, frame_size, 2, w_img, w_img], device=self.device)   # Generated displacement tensor
        seq_gen = torch.empty([batch_size, frame_size, 1, w_img, w_img], device=self.device)   # Generated sequence tensor

        for t in range(frame_size+1):
            idx_cur = t % frame_size           ### current index

            if t == 0:
                ### phi_x -----------------------------------------------------
                phi_x = self.net_phi_x(x0)

                ### net_infer -------------------------------------------------
                phi_x_h = torch.cat([phi_x, h_recurrent], 1)
                z_mu, z_logvar = self.net_infer(phi_x_h)

                ### z_t reparameterize ----------------------------------------
                z_t = self.zt_reparameterize(z_mu, z_logvar)

                ### phi_z -----------------------------------------------------
                phi_z = self.net_phi_z(z_t)

                ### x_pas -----------------------------------------------------
                x_pas = torch.clone(x0)

            elif t > 0:
                ### z_prior ---------------------------------------------------
                z_prior_mu, z_prior_logvar = self.net_z_prior(h_recurrent)

                ### z_t reparameterize ----------------------------------------
                z_t = self.zt_reparameterize(z_prior_mu, z_prior_logvar)

                ### phi_z -----------------------------------------------------
                phi_z = self.net_phi_z(z_t)

                ### net_gen ---------------------------------------------------
                phi_xp = self.net_phi_x(x_pas)
                phi_z_h = torch.cat([phi_z, phi_xp], 1)
                d_mu, d_logvar, d_log_v = self.net_gen(phi_z_h)

                ### d_t reparameterize ----------------------------------------
                d_cov = self.covariance_d(d_logvar, d_log_v)
                d_t = self.dt_reparameterize(d_mu, d_cov)

                ### x_out -----------------------------------------------------
                x_out = stn_2d(d_t[:, [1,0]], x_pas, self.device)

                ### phi_x -----------------------------------------------------
                phi_x = self.net_phi_x(x_out)
                x_pas = torch.clone(x_out)

                ### store results ---------------------------------------------
                dis_gen[:, idx_cur] = d_t
                seq_gen[:, idx_cur] = x_out

            ### net_recurrent -------------------------------------------------
            phi_z_x = torch.cat([phi_z, phi_x], 1)
            h_recurrent, hidden_state= self.net_recurrent(phi_z_x, hidden_state)
        return seq_gen, dis_gen

    ### generation from two frames ============================================
    def gen_two_frames(self, x0, x1, frame_size):
        batch_size, _, h_img, w_img = x0.shape

        ### Initialize the recurrent layer with zeros
        h_recurrent = torch.zeros([batch_size, 16, 32, 32], device=self.device)   # Recurrent layer state

        ### Initialize the hidden state of the ConvLSTM module
        hidden_state = self.ConvLSTM._init_hidden(batch_size=batch_size)   # Hidden state initialization

        ### Initialize empty tensors to store the Generated displacement and sequence
        dis_gen = torch.empty([batch_size, frame_size, 2, w_img, w_img], device=self.device)   # Generated displacement tensor
        seq_gen = torch.empty([batch_size, frame_size, 1, w_img, w_img], device=self.device)   # Generated sequence tensor

        for t in range(frame_size+1):
            idx_cur = t % frame_size           ### current index

            if t == 0:
                ### phi_x -----------------------------------------------------
                phi_x = self.net_phi_x(x0)

                ### net_infer -------------------------------------------------
                phi_x_h = torch.cat([phi_x, h_recurrent], 1)
                z_mu, z_logvar = self.net_infer(phi_x_h)

                ### z_t reparameterize ----------------------------------------
                z_t = self.zt_reparameterize(z_mu, z_logvar)

                ### phi_z -----------------------------------------------------
                phi_z = self.net_phi_z(z_t)

                ### x_pas -----------------------------------------------------
                x_pas = torch.clone(x0)

            elif t > 0:
                if t == 1:
                    ### phi_x -------------------------------------------------
                    phi_x = self.net_phi_x(x1)

                    ### net_infer ---------------------------------------------
                    phi_x_h = torch.cat([phi_x, h_recurrent], 1)
                    z_mu, z_logvar = self.net_infer(phi_x_h)

                    ### z_t reparameterize ------------------------------------
                    z_t = self.zt_reparameterize(z_mu, z_logvar)

                    ### phi_z -------------------------------------------------
                    phi_z = self.net_phi_z(z_t)

                    ### net_gen -----------------------------------------------
                    phi_xp = self.net_phi_x(x_pas)
                    phi_z_h = torch.cat([phi_z, phi_xp], 1)
                    d_mu, d_logvar, d_log_v = self.net_gen(phi_z_h)

                    ### d_t reparameterize ------------------------------------
                    d_cov = self.covariance_d(d_logvar, d_log_v)
                    d_t = self.dt_reparameterize(d_mu, d_cov)

                    ### x_out -------------------------------------------------
                    x_out = stn_2d(d_t[:, [1,0]], x_pas, self.device)

                    ### x_pas -------------------------------------------------
                    x_pas = torch.clone(x1)

                    ### store results -----------------------------------------
                    dis_gen[:, idx_cur] = d_t
                    seq_gen[:, idx_cur] = x_out

                else:
                    ### z_prior -----------------------------------------------
                    z_prior_mu, z_prior_logvar = self.net_z_prior(h_recurrent)

                    ### z_t reparameterize ------------------------------------
                    z_t = self.zt_reparameterize(z_prior_mu, z_prior_logvar)

                    ### phi_z -------------------------------------------------
                    phi_z = self.net_phi_z(z_t)

                    ### net_gen -----------------------------------------------
                    phi_xp = self.net_phi_x(x_pas)
                    phi_z_h = torch.cat([phi_z, phi_xp], 1)
                    d_mu, d_logvar, d_log_v = self.net_gen(phi_z_h)

                    ### d_t reparameterize ------------------------------------
                    d_cov = self.covariance_d(d_logvar, d_log_v)
                    d_t = self.dt_reparameterize(d_mu, d_cov)

                    ### x_out -------------------------------------------------
                    x_out = stn_2d(d_t[:, [1,0]], x_pas, self.device)

                    ### phi_x -------------------------------------------------
                    phi_x = self.net_phi_x(x_out)
                    x_pas = torch.clone(x_out)

                    ### store results -----------------------------------------
                    dis_gen[:, idx_cur] = d_t
                    seq_gen[:, idx_cur] = x_out

            ### net_recurrent -------------------------------------------------
            phi_z_x = torch.cat([phi_z, phi_x], 1)
            h_recurrent, hidden_state= self.net_recurrent(phi_z_x, hidden_state)
        return seq_gen, dis_gen