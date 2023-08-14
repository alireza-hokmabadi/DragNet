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

**Data Format:**  In this phase, the model utilizes 4-chamber LAX cine CMR images as input. The images are first cropped to a size of 128x128 pixels. Additionally, to manage computational costs, the frames are reduced to the specified frame size (default is 7). You can adjust this size based on your data and requirements. The dataset comprises 4620 samples with corresponding ground truth labels. Among these, 80% are allocated for training, while the remaining 20% are reserved for validation. To enhance speed, we package the data into a 5D tensor with dimensions of 4620x7x1x128x128 and load it into RAM.

**Note:** Due to copyright restrictions, we are unable to share the data. If you intend to use this code, ensure that your data has been preprocessed and cropped according to the specified criteria. Modify the data path as needed and adapt model parameters to suit your dataset.


- phase_a_train.py: You can run the code related to train phase using the following command on your local machine:
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
In this code, we have used 4-chamber LAX cine CMR images of UK-Biobank as input. First, we crop them in the size of 128*128 and also reduce the number of frames from 50 to the defined frame size (here 7 for default). total number of available samples with ground truth was 4620 and we used 80% of them for training and 20% for validation. Moreover to increase the run speed of the program, first we packed that data with a size of 460*7*1*128*128 and loaded it in the RAM.
However due to copy right rule, we cannot share the data and if you run this code, it raises the error of the data path. So, if you are going to use this code first prepare your data like this format and modify the data path or change the dataset and parameters of the model based on your needs.

- phase_b_test.py & phase_c_generation.py: You can run the code related to test and generation phase using the following command on your local machine:
```bash
python phase_b_test.py
python phase_c_generation.py
```
for this test and generation codes, we uploaded a sample sequence with the size of 1*7*1*128*128 and used it to show how the network works. you can modify the code by inserting your own data and parameters.


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


