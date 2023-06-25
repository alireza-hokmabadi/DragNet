import argparse



def parameter_parser():
    parser = argparse.ArgumentParser(description="Argument Parser")

    parser.add_argument('--frame_size', type=int, default=7 , help='Size of the frame, default value is 7')

    args = parser.parse_args()

    return args

    
    # parser.add_argument("-dir", dest ="<data path>",
    #                     default =  r"F:\Dataset Source Main\Img_Mask_Dataset_7_128_128\Data_LAX_Grt_4620_7_128_128", #data path
    #                     help = "Name of data's container. Default is data.")

    # parser.add_argument('-frame_siz', dest ="<int>", type=int, default=7 , help='Frame size, default is 7')
    # parser.add_argument('-epoch_siz', dest ="<int>", type=int, default=70 , help='Epoch size (train), default is 70')
    # parser.add_argument('-batch_siz', dest ="<int>", type=int, default=10 , help='Batch size, default is 10')

    # parser.add_argument('-zdim', dest ="<int>", type=int, default=64 , help='Latent (z) dim, default is 64.')
    # parser.add_argument('-sigblr', dest ="<float>", type=float, default=0.2 , help='Sigma gaussian filter (blur), default is 0.2.')

    # parser.add_argument('-lr', dest ="<float>", type=float, default=1e-3 , help='Learning rate, default is 1e-3.')




#     parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
#     parser.add_argument("--fixedmeshName", type=str, default='liv-mesh-9.vtk', required=True, help='Fixed mesh for registration, default is liver-Mesh-9.')
#     parser.add_argument('--batchsize', type=int, default=1 , help='Batch size, default is 1.')

    # parser.add_argument("--directory",
    #                     dest ="directory",
    #                     default =  '/usr/not-backed-up/scsk/Data/downsampled_vtk_4k_LV/', #benchmarks'   #C:/Users/skala/PycharmProjects/VGAE-PyTorch/LiverVGAE/liver3_decimated_0.9.vtp
 	  #                     #default =  '/usr/not-backed-up/scsk/Data/cpd_liver_mesh_139_dec0.9_0.6/vtk/',  ###livermesh_1024points  ####zcloud
    #               help = "Name of data's container. Default is data.")  ##'/content/drive/My Drive/Data/liver_mesh_simp/'

#     parser.add_argument("--Tdirectory",
#                         dest ="Tdirectory",
#                         # default =  '/content/drive/My Drive/Data/liver_mesh_simp/', #benchmarks'   #C:/Users/skala/PycharmProjects/VGAE-PyTorch/LiverVGAE/liver3_decimated_0.9.vtp
# 	                      default =  '/usr/not-backed-up/scsk/Data/Target/',  ### 1000215.vtu
#                   help = "Name of data's container. Default is data.")

#     parser.add_argument("--Resultdir",
#                         dest ="Resultdir",
#                         default = '/usr/not-backed-up/scsk/Code/Mesh_Reg_CPD/results/', #benchmarks'   #C:/Users/skala/PycharmProjects/VGAE-PyTorch/LiverVGAE/liver3_decimated_0.9.vtp
# 	                help = "Results directory.") ##'/content/drive/MyDrive/code/LiverVGAE/results/'

#     # parser.add_argument("--Modeldir",
#     #                     dest ="Modeldir",
#     #                     default = '/content/drive/MyDrive/code/VAE_Hyper/saved_models/', #benchmarks'   #C:/Users/skala/PycharmProjects/VGAE-PyTorch/LiverVGAE/liver3_decimated_0.9.vtp
# 	  #               help = "Saved model directory.") ## '/content/drive/MyDrive/code/LiverVGAE/saved_models/'
#     parser.add_argument("--test_size",
#                         dest = "test_size",
#                         type = float,
#                         default = 0.10,
# 	                help = "Size of test dataset. Default is 10%.")

    # return parser.parse_args()
