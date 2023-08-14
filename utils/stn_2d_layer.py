
import torch
import torch.nn.functional as F

### generate_grid_2d ==========================================================
def generate_grid_2d(B, C, H, W, device):
    aff = torch.FloatTensor([[[1, 0, 0],[0, 1, 0]]])
    aff = aff.expand(B, 2, 3)  # expand to the number of batches you need
    grid = torch.nn.functional.affine_grid(aff, size=(B,C,H,W)).to(device)
    return grid

### stn_2d ====================================================================
def stn_2d(flow, img, device):
    B, C, H, W = flow.shape
    grid = generate_grid_2d(B, C, H, W, device)

    factor = torch.FloatTensor([[[[2/W, 2/H]]]]).to(device)
    deformation = flow.permute(0,2,3,1)*factor + grid
    warped_img = F.grid_sample(img, deformation, align_corners=False)
    return warped_img