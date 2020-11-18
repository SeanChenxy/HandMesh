import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import openmesh as om
from os import path as osp
from utils import utils, mesh_sampling
from psbody.mesh import Mesh
import pickle


def read_mesh(path):
    mesh = om.read_trimesh(path)
    face = torch.from_numpy(mesh.face_vertex_indices()).T.type(torch.long)
    x = torch.tensor(mesh.points().astype('float32'))
    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    edge_index = to_undirected(edge_index)
    return Data(x=x, edge_index=edge_index, face=face)


def save_mesh(fp, x, f):
    om.write_mesh(fp, om.TriMesh(x, f))


def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()


def spiral_tramsform(transform_fp, template_fp, ds_factors, seq_length, dilation):
    if not osp.exists(transform_fp):
        print('Generating transform matrices...')
        mesh = Mesh(filename=template_fp)
        # ds_factors = [3.5, 3.5, 3.5, 3.5]
        _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
            mesh, ds_factors)
        tmp = {
            'vertices': V,
            'face': F,
            'adj': A,
            'down_transform': D,
         'up_transform': U
        }

        with open(transform_fp, 'wb') as fp:
            pickle.dump(tmp, fp)
        print('Done!')
        print('Transform matrices are saved in \'{}\''.format(transform_fp))
    else:
        with open(transform_fp, 'rb') as f:
            tmp = pickle.load(f, encoding='latin1')

    spiral_indices_list = [
        utils.preprocess_spiral(tmp['face'][idx], seq_length[idx], tmp['vertices'][idx], dilation[idx])#.to(device)
        for idx in range(len(tmp['face']) - 1)
    ]

    down_transform_list = [
        utils.to_sparse(down_transform)#.to(device)
        for down_transform in tmp['down_transform']
    ]
    up_transform_list = [
        utils.to_sparse(up_transform)#.to(device)
        for up_transform in tmp['up_transform']
    ]

    return spiral_indices_list, down_transform_list, up_transform_list, tmp


if __name__ == '__main__':
    mesh = read_mesh('../data/FreiHAND/template/template.obj')
    save_mesh('../data/FreiHAND/template/template.obj', mesh.x.numpy(), mesh.face.numpy().T)
