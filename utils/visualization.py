
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio as iio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

### visualization =============================================================
def visualization(x_target, x_prediction, displacement, label_title= "Registration", label_out= "Prediction", edg= 2):
    frame_size, h_img, w_img = x_target.shape

    fig, axs = plt.subplots(4, frame_size, figsize=(19,9.5))
    fig.suptitle(label_title, fontweight="bold", fontsize= 14)

    for i in range(frame_size):
        ### x_target ----------------------------------------------------------
        ax = axs[0,i]
        ax.imshow(x_target[i], cmap= "gray", vmin=0, vmax=255)

        if i == 0:
            ax.set_title("0 (ED)")
            ax.set_ylabel("Original")
        elif i == 3:
            ax.set_title("3 (ES)")
        else:
            ax.set_title(str(i))

        ax.set_xticks([])
        ax.set_yticks([])

        ### x_prediction ------------------------------------------------------
        ax = axs[1,i]
        ax.imshow(x_prediction[i], cmap= "gray", vmin=0, vmax=255)

        if i == 0:
            ax.set_ylabel(label_out)

        diff_rmse = np.sqrt(np.mean(((x_target[i, edg:-edg, edg:-edg] - x_prediction[i, edg:-edg, edg:-edg])/255.)**2))
        ax.set_title(f"RMSE= {diff_rmse :.03f}")

        ax.set_xticks([])
        ax.set_yticks([])

        ### displacement ------------------------------------------------------
        dis_x = displacement[i, 0]
        dis_y = displacement[i, 1]
        dis_colorwheel = compute_color(dis_x, dis_y) / 255.0

        ax = axs[2,i]
        ax.imshow(dis_colorwheel, vmin=0, vmax=1)

        if i == 0:
            ax.set_ylabel("DVF Maps")
            ax.get_yaxis().set_label_coords(-0.06,0.5)

        ax.set_xticks([])
        ax.set_yticks([])

        ### jacobian_det ------------------------------------------------------
        colors1 = plt.cm.gray(np.linspace(1.0, 0.0, 128))
        colors2 = plt.cm.cool(np.linspace(0., 1, 128))
        colors = np.vstack((colors1, colors2))
        mymap = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)

        jac_det_calc = Fn_jac_det_calc_2d(dis_x, dis_y)

        ax = axs[3,i]
        pcm = ax.imshow(jac_det_calc, cmap=mymap, vmin=-2, vmax=2)

        divider = make_axes_locatable(ax)

        if i == frame_size-1:
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pcm, cax= cax)

        if i == 0:
            ax.set_ylabel("Jac. Det.")

        ax.axis([0, w_img, 0, h_img])
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

    ### colorwheel key --------------------------------------------------------
    ax = axs[2,frame_size-1]
    axins = inset_axes(ax, 0.8, 0.8, loc= "right", bbox_to_anchor= (1.7,0.65), bbox_transform= ax.transAxes)
    colorwheel_key = iio.imread("./utils/color_wheel_key.png")

    axins.imshow(colorwheel_key, vmin=0, vmax=1)
    axins.set_title("colorwheel key", fontsize= 10)
    axins.set_xticks([])
    axins.set_yticks([])

    # fig.tight_layout(pad=0.2)
    fig.align_labels()
    plt.show()

### ===========================================================================
def make_colorwheel():
    """
    Generates a colorwheel for optical flow visualization.

    Returns:
        np.ndarray: A colorwheel represented as an array of RGB values.
    """
    RY, YG, GC, CB, BM, MR = [15, 6, 4, 11, 13, 6]

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3), dtype=np.uint8)  # r g b

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY, 1) / RY)
    col += RY

    # YG
    colorwheel[col:YG + col, 0] = 255 - np.floor(255 * np.arange(0, YG, 1) / YG)
    colorwheel[col:YG + col, 1] = 255
    col += YG

    # GC
    colorwheel[col:GC + col, 1] = 255
    colorwheel[col:GC + col, 2] = np.floor(255 * np.arange(0, GC, 1) / GC)
    col += GC

    # CB
    colorwheel[col:CB + col, 1] = 255 - np.floor(255 * np.arange(0, CB, 1) / CB)
    colorwheel[col:CB + col, 2] = 255
    col += CB

    # BM
    colorwheel[col:BM + col, 2] = 255
    colorwheel[col:BM + col, 0] = np.floor(255 * np.arange(0, BM, 1) / BM)
    col += BM

    # MR
    colorwheel[col:MR + col, 2] = 255 - np.floor(255 * np.arange(0, MR, 1) / MR)
    colorwheel[col:MR + col, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    Compute color image from optical flow vectors u and v.

    Parameters:
        u (np.ndarray): Optical flow vector in x-direction
        v (np.ndarray): Optical flow vector in y-direction

    Returns:
        np.ndarray: Color image with optical flow visualization
    """
    colorwheel = make_colorwheel()

    # Identify NaN values in u and v arrays
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)

    # Replace NaN values with zero in u and v arrays
    u[nan_u] = 0
    v[nan_v] = 0

    ncols = colorwheel.shape[0]
    radius = np.hypot(u, v)
    angle = np.arctan2(u, v) / np.pi
    fk = (angle + 1) / 2 * (ncols - 1)  # -1~1 mapped to 1~ncols
    k0 = fk.astype(np.uint8)            # 1, 2, ..., ncols
    k1 = (k0 + 1) % ncols               # Modulo operation to handle edge case
    f = fk - k0

    img = np.empty((k1.shape[0], k1.shape[1], 3), dtype=np.uint8)  # Initialize as uint8 array for better performance
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx] * (1 - col[idx])  # increase saturation with radius
        col[~idx] *= 0.75                            # out of range
        img[:, :, 2 - i] = (255 * col).astype(np.uint8)  # Use astype() for type casting

    return img

### jacobian_2d ===============================================================
def Fn_jac_det_calc_2d(dis_x, dis_y, add_identity=True):
    """Computes jacobian_2d of given deformation phi = [dis_x, dis_y]."""
    gx_y, gx_x = np.gradient(dis_x)
    gy_y, gy_x = np.gradient(dis_y)
    if add_identity:
        gx_x += 1.
        gy_y += 1.

    det = gx_x * gy_y - gy_x * gx_y
    return det