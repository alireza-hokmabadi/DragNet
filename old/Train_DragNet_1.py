import torch
import torch.utils
import torch.utils.data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import pickle
from time import time
import numpy as np
import os

# from src import VTKObject,writeObjects, image_loader, Create_unstructuredGrid, Create_pointcloud, clip_by_tensor , Convert_Adj_EdgeList, convert_Edg_to_face
from src import parameter_parser

print(chr(12))

if __name__ == '__main__':
    args = parameter_parser()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device is :', device)

    print('Frame size:', args.frm_siz)
    print('lr_rate:', args.lr_rate)
    # meshdata = os.listdir(args.directory)
    # start_time = time.time()

    # for ind, meshName in enumerate(meshdata):

    #   ###############################################
    #   #        Moving Mesh
    #   ###############################################
    #   # Num: 96 --Patient: ['liv-mesh-58.vtu']  ## meshName[0]='liv-mesh-58.vtu'
    #   print("Num:", ind, "-- Moving Patient: {}".format(meshName))
    #   path = os.path.join(args.directory, meshName)
    #   print(path)
    #   MeshReg(args,meshName)

    # print("--- %s seconds ---" % (time.time() - start_time))