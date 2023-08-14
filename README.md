# DragNet (Deformable Registration and Generative Network)

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Description
Pytorch implementation of DragNet (Deformable Registration and Generative Network).

This repository is a torch implementation of DragNet based on "DragNet: Learning-based deformable registration for realistic cardiac MR sequence generation from a single frame"
DOI: https://doi.org/10.1016/j.media.2022.102678

![alt text](https://github.com/alireza-hokmabadi/DragNet/blob/master/data/model_structure.jpg)

### Installation
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
You can run the code related to train phase using the following command on your local machine:
```bash
python phase_a_train.py [-h] [--data_path] [--frame_size] [--sigma_blur] [--epoch_size] [--batch_size] [--learning_rate]

optional arguments:
  -h, --help        show this help message and exit
  --data_path       Data path
  --frame_size      Frame size (default: 7)
  --sigma_blur      Sigma blur (Standard deviation for Gaussian kernel, default: 0.2)
  --epoch_size      Epoch size (default: 70)
  --batch_size      Batch size (default: 10)
  --learning_rate   Learning rate (default: 0.001)
```

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

## contact
For any comment, please contact Alireza Hokmabadi
- Email: [a.hokmabadi.ee@gmail.com](mailto:a.hokmabadi.ee@gmail.com)


