
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from scipy import ndimage
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from utils.parameter_parser import parse_args
from utils.dragnet_model import dragnet

# np.random.seed(1234)

### ===========================================================================
### Dataset ===================================================================
class Dataset(Dataset):
    def __init__(self, data, sigma_blur):
        self.data = data
        self.sigma_blur = sigma_blur

    def __getitem__(self, index):
        seq = self.data[index]

        ### apply Gaussian filter to have smooth deformations
        seq_blur = ndimage.gaussian_filter(seq.astype(np.float64), self.sigma_blur, mode="constant")

        ### normalize
        seq_blur /= 255.0

        ### plot the whole sequence
        # n_frm = seq.shape[0]
        # fig, axs = plt.subplots(2,n_frm)

        # for nf in range(n_frm):
        #     ax =  axs[0, nf]
        #     ax.imshow(seq[nf], cmap= "gray")
        #     ax.set_title(f"seq_in [{nf}]")
        #     ax.axis("off")

        #     ax =  axs[1, nf]
        #     ax.imshow(seq_blur[nf], cmap= "gray")
        #     ax.set_title(f"seq_blur [{nf}]")
        #     ax.axis("off")

        # fig.tight_layout()
        # plt.show()

        ### convert to tensor
        seq_tensor = torch.tensor(seq_blur).unsqueeze(1).float()

        return seq_tensor

    def __len__(self):
        return len(self.data)

### ===========================================================================
### Main ======================================================================
if __name__ == "__main__":

    ### parameters ============================================================
    args = parse_args()
    # print("Frame size:", args.frame_size)

    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ### load data =============================================================
    with open(args.data_path, "rb") as f:
        input_data = pickle.load(f)

    img_seq_pack = input_data["img_seq_pack"]     # shape: (4620, 7, 128, 128), dtype("uint8"), Min:0, Max:255

    ### call dataset ==========================================================
    partition = {}
    partition["train"], partition["val"] = train_test_split(np.arange(len(img_seq_pack)), test_size=0.2, random_state=1234)

    data_train = img_seq_pack[partition["train"]]
    data_val = img_seq_pack[partition["val"]]

    dataset_train = Dataset(data_train, args.sigma_blur)
    dataloader_train = DataLoader(dataset_train, batch_size= args.batch_size, shuffle= True)

    dataset_val = Dataset(data_val, args.sigma_blur)
    dataloader_val = DataLoader(dataset_val, batch_size= args.batch_size, shuffle= False)

    n_batch_train = len(dataloader_train)  # Total number of batches for the train phase
    n_batch_val = len(dataloader_val)      # Total number of batches for the validation phase

    ### call model ============================================================
    model = dragnet(device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr= args.learning_rate, amsgrad=True)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 5, gamma= 0.5, last_epoch=-1)
    writer = SummaryWriter()

    print(model)
    print("number of parameters: ", sum([param.numel() for param in model.parameters()]))

    ### run model (train and validation) ======================================
    for epochs in range(args.epoch_size):

        ### train =============================================================
        model.train()
        for batch_idx, seq_in in enumerate(dataloader_train):
            seq_in = Variable(seq_in).to(device)

            optimizer.zero_grad()
            seq_out, dis_out, mse_loss, klz_loss, smooth_loss, kld_loss, total_loss = model(seq_in)

            total_loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)       # Clip gradients to prevent exploding gradients

            # Extract loss values from tensors
            mse_loss_val = mse_loss.item()
            klz_loss_val = klz_loss.item()
            smooth_loss_val = smooth_loss.item()
            kld_loss_val = kld_loss.item()
            total_loss_val = total_loss.item()

            # Log loss values for training using tensorboard writer
            counter_idx_train = epochs * n_batch_train + batch_idx                      # counter index train
            writer.add_scalar("train_mse_loss", mse_loss_val, counter_idx_train)        # Log MSE loss
            writer.add_scalar("train_klz_loss", klz_loss_val, counter_idx_train)        # Log KL divergence loss z (latent spase)
            writer.add_scalar("train_smooth_loss", smooth_loss_val, counter_idx_train)  # Log smoothness loss
            writer.add_scalar("train_kld_loss", kld_loss_val, counter_idx_train)        # Log KL divergence loss d (displacement)
            writer.add_scalar("train_total_loss", total_loss_val, counter_idx_train)    # Log total loss

            # Print training progress every 5 batches
            if batch_idx % 5 == 0:
                print(f"Epoch: {epochs} [{batch_idx}/{n_batch_train} ({batch_idx/n_batch_train*100.0 :.0f}%)]"
                      f"\t mse_loss: {mse_loss_val :.5f}"
                      f"\t klz_loss: {klz_loss_val :.5f}"
                      f"\t smooth_loss: {smooth_loss_val :.5f}"
                      f"\t kld_loss: {kld_loss_val :.5f}"
                      f"\t total_loss: {total_loss_val :.5f}")

        ### learning rate update ==============================================
        # lr_scheduler.step()

        ### validation ========================================================
        model.eval()
        with torch.no_grad():

            for batch_idx, seq_in in enumerate(dataloader_val):
                seq_in = seq_in.to(device)

                seq_out, dis_out, mse_loss, klz_loss, smooth_loss, kld_loss, total_loss = model(seq_in)

                # Extract loss values from tensors
                mse_loss_val = mse_loss.item()
                klz_loss_val = klz_loss.item()
                smooth_loss_val = smooth_loss.item()
                kld_loss_val = kld_loss.item()
                total_loss_val = total_loss.item()

                # Log loss values for validation using tensorboard writer
                counter_idx_val = epochs * n_batch_val + batch_idx                      # counter index val
                writer.add_scalar("val_mse_loss", mse_loss_val, counter_idx_val)        # Log MSE loss
                writer.add_scalar("val_klz_loss", klz_loss_val, counter_idx_val)        # Log KL divergence loss z (latent spase)
                writer.add_scalar("val_smooth_loss", smooth_loss_val, counter_idx_val)  # Log smoothness loss
                writer.add_scalar("val_kld_loss", kld_loss_val, counter_idx_val)        # Log KL divergence loss d (displacement)
                writer.add_scalar("val_total_loss", total_loss_val, counter_idx_val)    # Log total loss

                # Print validation progress every 5 batches
                if batch_idx % 5 == 0:
                    print(f"Validation phase ===> Epoch: {epochs} [{batch_idx}/{n_batch_val} ({batch_idx/n_batch_val*100.0 :.0f}%)]"
                          f"\t mse_loss: {mse_loss_val :.5f}"
                          f"\t klz_loss: {klz_loss_val :.5f}"
                          f"\t smooth_loss: {smooth_loss_val :.5f}"
                          f"\t kld_loss: {kld_loss_val :.5f}"
                          f"\t total_loss: {total_loss_val :.5f}")

### save weights ==============================================================
torch.save(model.state_dict(), f"./weights/model_weights_epoch_{epochs}.pth")
writer.close()

print(" Well-done! ".center(50,"="))