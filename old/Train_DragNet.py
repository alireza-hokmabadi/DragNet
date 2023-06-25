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
# from src import parameter_parser



#################################################################################################################
# Main function for training
def MeshReg(args,meshName):
        start_time = time.time()

        path = os.path.join(args.directory, meshName)


        data = VTKObject(filename=path)
        faces = data.triangles
        faces = np.int32(faces)


            ###############################################
            #        Fixed Mesh
            ###############################################
        pathT = os.path.join(args.Tdirectory, args.fixedmeshName)
        dataT = VTKObject(filename=pathT)

        facesT = dataT.triangles
        facesT = np.int32(facesT)
        base = os.path.basename(meshName)
        Name = os.path.splitext(base)[0]
        print(Name)

            ###############################################
            #        Mesh Registration
            ###############################################
        print('X: TARGET:', dataT.points.shape)
        print('Y: MOVING:', data.points.shape)

        ############################ cpd  #########################################################################
        '''
        Y : MOVING : Mx3

        X : TARGET : Nx3

        '''
        #print(dataT.points)
        #print(data.points)
        #reg = DeformableRegistration(**{'X': dataT.points, 'Y': data.points})
        #TY, _ = reg.register()
        reg = RigidRegistration(**{'X': dataT.points, 'Y': data.points})
        # reg = reg.to(device=device)

        TY, (s_reg, R_reg, t_reg) = reg.register()

        if np.isfinite(TY).all() == 'True':
            print('input has NaN:', np.isfinite(TY).all())
        if np.isfinite(t_reg).all() == 'True':
            print('input has NaN:', np.isfinite(t_reg).all())
        if np.isfinite(s_reg).all() == 'True':
            print('input has NaN:', np.isfinite(s_reg).all())
        if np.isfinite(R_reg).all() == 'True':
            print('input has NaN:', np.isfinite(R_reg).all())
        #a = EMRegistration(**{'X': dataT.points, 'Y': data.points})
        P = reg.P
#        print(P)
#        print(reg.X)
#        print(reg.Y)

        print('Corresp map:',P.shape) ## MxN
        #print(P)
        print('TY: MOVED:', TY.shape) ## Mx3


#        print(1/sum(P))

#        print('Normalized h_CPD: MOVED:', h_CPD.shape)
        #print(reg.Np)
        print(np.sum(P, axis=0))

        #den = np.sum(P, axis=1)
        #print(den)
        #P = np.divide(P, den)
        ##print(np.argmax(P, axis=1)) ## Confidence Map
        h_CPD = np.matmul(np.transpose(P),TY) ## Nx3

        print('h_CPD: MOVED:', h_CPD.shape)

        #print('transform_point_cloud:',Tpoint)
        torch.save(P, args.Resultdir + 'P/P_{}.pt' .format(Name))

        plt.figure()
        shw = plt.imshow(P)
        bar = plt.colorbar(shw)
        plt.xlabel('Atlas Shape')
        plt.ylabel('Observed Shape')
        bar.set_label('ColorBar')
        plt.savefig(args.Resultdir+'P_fig/P_{}.png'.format(Name),dpi=1200)



        ###assert_array_almost_equal(TY, X, decimal=0)


        ### write as vtk ==========================================================


        mesh_cpd = tvtk.PolyData(points=TY, polys=faces )
        write_data(mesh_cpd, args.Resultdir + 'h_CPD/CPD_' + Name+ ".vtk")

#        mesh_cpd_T = tvtk.PolyData(points=h_CPD, polys=facesT )
#        write_data(mesh_cpd_T, args.Resultdir + 'h_CPD/h_CPD_' + Name+ ".vtk")

        print("--- %s seconds ---" % (time.time() - start_time))
        print('################################################')
        return



if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device is :', device)

    args = parameter_parser()

    print('Data Dir:', args.directory)
    print('Result dir:', args.Resultdir)
    meshdata = os.listdir(args.directory)
    start_time = time.time()

    for ind, meshName in enumerate(meshdata):

      ###############################################
      #        Moving Mesh
      ###############################################
      # Num: 96 --Patient: ['liv-mesh-58.vtu']  ## meshName[0]='liv-mesh-58.vtu'
      print("Num:", ind, "-- Moving Patient: {}".format(meshName))
      path = os.path.join(args.directory, meshName)
      print(path)
      MeshReg(args,meshName)

    print("--- %s seconds ---" % (time.time() - start_time))