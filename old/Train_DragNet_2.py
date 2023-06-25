import torch

from src import parse_args

if __name__ == '__main__':
    args = parse_args.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # parser.add_argument('-frame_siz', dest ="<int>", type=int, default=7 , help='Frame size, default is 7')
    # parser.add_argument('-epoch_siz', dest ="<int>", type=int, default=70 , help='Epoch size (train), default is 70')
    # parser.add_argument('-batch_siz', dest ="<int>", type=int, default=10 , help='Batch size, default is 10')

    print('Frame size:', args.frame_siz)
    # print('lr_rate:', args.lr_rate)
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