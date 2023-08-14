
import torch

### loss function =============================================================
def loss_fn(x_target, x_prediction, z_mu, z_logvar, z_prior_mu, z_prior_logvar, d_t, d_mu, d_cov):

    ### MSE loss --------------------------------------------------------------
    sigma_mse = 1.0
    mse_loss_t = 1.0 / sigma_mse**2 * torch.mean(torch.square(x_target - x_prediction))

    ### KL divergence loss z --------------------------------------------------
    klz_element =  z_prior_logvar - z_logvar - 1. + (z_logvar.exp() + (z_mu - z_prior_mu).pow(2)) / (z_prior_logvar.exp()+1e-6)
    klz_loss_t = 0.5 * torch.mean(klz_element)

    ### smoothness loss -------------------------------------------------------
    dif_x = diff_fun(d_t, 2)
    dif_y = diff_fun(d_t, 3)

    n_dim = torch.Tensor.dim(d_t)
    dif_x_mean = torch.mean(dif_x, dim=[*range(1, n_dim)])
    dif_y_mean = torch.mean(dif_y, dim=[*range(1, n_dim)])
    smooth_loss_t = (torch.mean(dif_x_mean) + torch.mean(dif_y_mean)) / 2.

    ### KL divergence loss d --------------------------------------------------
    det_d_cov = d_cov[:,:,:,0,0]*d_cov[:,:,:,1,1] - d_cov[:,:,:,0,1]*d_cov[:,:,:,1,0]
    d_mu_expand = d_mu.permute(0,2,3,1).unsqueeze(-1)

    kld_element_1 = -torch.log(det_d_cov)
    kld_element_2 = d_cov[:,:,:,0,0] + d_cov[:,:,:,1,1]
    kld_element_3 = torch.matmul(d_mu_expand.permute(0,1,2,4,3), d_mu_expand).squeeze(-1).squeeze(-1)

    kld_element = kld_element_1 - 2.0 +  kld_element_2 + kld_element_3
    kld_loss_t = 0.5 * torch.mean(kld_element)

    return mse_loss_t, klz_loss_t, smooth_loss_t, kld_loss_t

def diff_fun(y_in, k_dim):
    n_dim = torch.Tensor.dim(y_in)

    rp = [k_dim, *range(k_dim), *range(k_dim + 1, n_dim)]
    y = y_in.permute(rp)

    df = y[1:, ...] - y[:-1, ...]

    rn = [*range(1, k_dim+1), 0, *range(k_dim+1, n_dim)]
    df = df.permute(rn)  # permute back

    df = df.pow(2)
    return df