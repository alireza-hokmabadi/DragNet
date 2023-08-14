
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import ndimage
import torch
from torch.utils.data import Dataset, DataLoader
import itertools

from utils.dragnet_model import dragnet
from utils.visualization import visualization

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
    frame_size = 7
    sigma_blur = 0.2

    saved_weights = "./weights/model_weights_epoch_70.pth"
    data_path = "./utils/test_sample"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ### load data =============================================================
    with open(data_path, "rb") as f:
        input_data = pickle.load(f)

    deta_test = input_data["test_seq"]     # shape: (1, 7, 128, 128), dtype("uint8"), Min:0, Max:255

    ### call dataset ==========================================================
    dataset_test = Dataset(deta_test, sigma_blur)
    dataloader_test = DataLoader(dataset_test, batch_size= 1, shuffle= False)

    ### call model ============================================================
    model = dragnet(device).to(device)
    model.load_state_dict(torch.load(saved_weights, map_location=device))      # load the saved weights

    print(model)
    print("number of parameters: ", sum([param.numel() for param in model.parameters()]))

    ### run model (test) ======================================================
    seq_in = next(itertools.islice(dataloader_test, 0, None))

    model.eval()
    with torch.no_grad():
        seq_in = seq_in.to(device)
        seq_out, dis_out, _, _, _, _, _ = model(seq_in)

    ### Extract and prepare visualizations of the target and predicted sequences and displacement
    x_target = seq_in[0, :, 0].cpu().detach().numpy() * 255.0       # Extract and scale the target sequence
    x_prediction = seq_out[0, :, 0].cpu().detach().numpy() * 255.0  # Extract and scale the predicted sequence
    displacement = dis_out[0, :].cpu().detach().numpy()             # Extract the displacement map

    visualization(x_target, x_prediction, displacement, label_title= "Spatio-temporal registration (test phase)", label_out= "Prediction", edg= 2)

print(" Well-done! ".center(50,"="))