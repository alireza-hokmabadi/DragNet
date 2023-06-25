import torch

from src import parameter_parser

if __name__ == '__main__':
    args = parameter_parser.parse_args()
    # print('Frame size:', args.frame_size)

    for arg_name, arg_value in vars(args).items():
        print(f'{arg_name}: {arg_value}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)




    # print('Frame size:', args.frame_siz)
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