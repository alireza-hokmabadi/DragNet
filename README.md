# DragNet (Deformable Registration and Generative Network)

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

## Description
Pytorch implementation of DragNet (Deformable Registration and Generative Network).

This repository is a torch implementation of DragNet based on "DragNet: Learning-based deformable registration for realistic cardiac MR sequence generation from a single frame".

The paper is open access at https://doi.org/10.1016/j.media.2022.102678 or you can download it [here](https://github.com/alireza-hokmabadi/DragNet/blob/master/data/DragNet_paper.pdf)

**Model structure:**
<div align="center" style="margin-bottom: 40px;">
  <img src="https://github.com/alireza-hokmabadi/DragNet/blob/master/data/model_structure.jpg" alt="Image 1" width="80%">
</div>

**Spatio-temporal registration:**
<div align="center" style="margin-bottom: 40px;">
  <img src="https://github.com/alireza-hokmabadi/DragNet/blob/master/data/spatio_temporal_registration.jpg" alt="Image 2" width="80%">
</div>

**Generated sample:**

<div align="center" style="margin-bottom: 40px;">
  <img src="https://github.com/alireza-hokmabadi/DragNet/blob/master/data/Generated_sequence_0.gif" alt="Image 3" width="25%">
</div>



## Installation
To use this project, follow these steps:

1. Clone the Repository: Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/alireza-hokmabadi/DragNet.git
```

2. Navigate to the Project Directory: Change your working directory to the project folder:

```bash
cd DragNet
```

3. Install Dependencies: Install the project's dependencies listed in the requirements.txt file:

```bash
pip install -r requirements.txt
```

## Usage

### Training Phase (phase_a_train.py)

To train the model, follow these steps:

**Run Training Code:** Open a terminal window and navigate to the project directory. Run the training code using the following command:

```bash
python phase_a_train.py [--data_path] [--frame_size] [--sigma_blur] [--epoch_size] [--batch_size] [--learning_rate]
```

Optional arguments:

- `--data_path`: Data path
- `--frame_size`: Frame size (default: 7)
- `--sigma_blur`: Sigma blur (Standard deviation for Gaussian kernel, default: 0.2)
- `--epoch_size`: Epoch size (default: 70)
- `--batch_size`: Batch size (default: 10)
- `--learning_rate`: Learning rate (default: 0.001)

Example:

```bash
python phase_a_train.py --data_path /path/to/your/data
```

**Data Format:**  The model utilizes 4-chamber LAX cine CMR images as input. The images are first cropped to a size of 128x128 pixels. Additionally, to manage computational costs, the frames are reduced to the specified frame size (default is 7). You can adjust this size based on your data and requirements. The dataset comprises 4620 samples with corresponding ground truth labels. Among these, 80% are allocated for training, while the remaining 20% are reserved for validation. To enhance speed, we package the data into a 5D tensor with dimensions of 4620x7x1x128x128 and load it into RAM.

**Note:** Due to copyright restrictions, we are unable to share the data. If you intend to use this code, ensure that your data has been preprocessed and cropped according to the specified criteria. Modify the data path as needed and adapt model parameters to suit your dataset.



### Test and Generation Phases (phase_b_test.py & phase_c_generation.py)

To perform testing and sequence generation, follow these steps:

**Run Test Code:**  Open a terminal window and navigate to the project directory. Run the test code using the following command:

```bash
python phase_b_test.py
```

**Run Generation Code:**  Similarly, run the sequence generation code using the following command:

```bash
python phase_c_generation.py
```

For these test and generation codes, we provide a sample sequence with dimensions of 1x7x1x128x128 to demonstrate the network's functionality. You can modify the code to incorporate your own data and adjust parameters as needed.


## Citation

If you are interested in our paper or used our codes, please cite it as:

```bibtex
@article{zakeri2023dragnet,
  title={DragNet: learning-based deformable registration for realistic cardiac MR sequence generation from a single frame},
  author={Zakeri, A. and Hokmabadi, A. and Bi, N. and Wijesinghe, I. and Nix, M. G. and Petersen, S. E. and Frangi, A. F. and Taylor, Z. A. and Gooya, A.},
  journal={Medical Image Analysis},
  volume={83},
  pages={102678},
  year={2023},
}
```

If you have any questions, contact me by this email: [a.hokmabadi.ee@gmail.com](mailto:a.hokmabadi.ee@gmail.com)