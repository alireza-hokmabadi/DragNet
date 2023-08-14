import argparse

def parse_args():

    description = """
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    Implementation of Deformable Registration and Generative Network (DragNet)
    Based on "DragNet: Learning-based deformable registration for realistic cardiac MR sequence generation from a single frame"
    DOI: https://doi.org/10.1016/j.media.2022.102678
    For any comment, please contact Alireza Hokmabadi (a.hokmabadi.ee@gmail.com)
    * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    """

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    optional_args = parser._action_groups.pop()

    # required_args = parser.add_argument_group("required arguments")
    # required_args.add_argument("--data_path", metavar='', type=str, help="Data path", required=True)
    optional_args.add_argument("--data_path", metavar='', type=str, default=r"F:\Dataset Source Main\Img_Mask_Dataset_7_128_128\Data_LAX_Grt_4620_7_128_128_dict", help="Data path")

    optional_args.add_argument("--frame_size", metavar='', type=int, default=7, help="Frame size (default: 7)")
    optional_args.add_argument("--sigma_blur", metavar='', type=float, default=0.2, help="Sigma blur (Standard deviation for Gaussian kernel, default: 0.2)")
    optional_args.add_argument("--epoch_size", metavar='', type=int, default=70, help="Epoch size (default: 70)")
    optional_args.add_argument("--batch_size", metavar='', type=int, default=10, help="Batch size (default: 10)")
    optional_args.add_argument("--learning_rate", metavar='', type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser._action_groups.append(optional_args)

    # parser.print_help()

    args = parser.parse_args()
    return args
