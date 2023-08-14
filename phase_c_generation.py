
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

    ### run model (generation phase) ==========================================
    seq_in = next(itertools.islice(dataloader_test, 0, None))
    frame_0 = seq_in[:, 0]
    frame_1 = seq_in[:, 1]

    model.eval()
    with torch.no_grad():
        frame_0 = frame_0.to(device)
        frame_1 = frame_1.to(device)

        ### generation from a single frame ------------------------------------
        seq_gen_0, dis_gen_0 = model.gen_one_frame(frame_0, frame_size)

        ### generation from two frames ----------------------------------------
        seq_gen_1, dis_gen_1 = model.gen_two_frames(frame_0, frame_1, frame_size)

    ### Extract and prepare visualizations of the input and generated sequences and displacement
    x_input = seq_in[0, :, 0].cpu().detach().numpy() * 255.0       # Extract and scale the input sequence

    x_generated_0 = seq_gen_0[0, :, 0].cpu().detach().numpy() * 255.0  # Extract and scale the generated sequence from a single frame
    displacement_0 = dis_gen_0[0, :].cpu().detach().numpy()            # Extract the displacement map from a single frame

    x_generated_1 = seq_gen_1[0, :, 0].cpu().detach().numpy() * 255.0  # Extract and scale the generated sequence from two frames
    displacement_1 = dis_gen_1[0, :].cpu().detach().numpy()            # Extract the displacement map from two frames


    visualization(x_input, x_generated_0, displacement_0, label_title= "Generation #0 (from a single frame)", label_out= "Generation #0", edg= 2)
    visualization(x_input, x_generated_1, displacement_1, label_title= "Generation #1 (from two frames)", label_out= "Generation #1", edg= 2)

print(" Well-done! ".center(50,"="))