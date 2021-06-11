import torch
import numpy as np


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    return np.stack((x, y, z), 1)


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
    return cam_coord


def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1, 3)).transpose(1, 0)).transpose(1, 0)
    return world_coord


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s)

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2


def align_sc_tr(A, B):
    """ Align the 3D joint location with the ground truth by scaling and translation """

    predCurr = A.copy()
    # normalize the predictions
    s = np.sqrt(np.sum(np.square(predCurr[4] - predCurr[10])))
    if s>0:
        predCurr = predCurr / s

    # get the scale of the ground truth
    sGT = np.sqrt(np.sum(np.square(B[9] - B[10])))

    # make predictions scale same as ground truth scale
    predCurr = predCurr * sGT

    # make preditions translation of the wrist joint same as ground truth
    predCurrRel = predCurr - predCurr[0:1, :]
    preds_sc_tr_al = predCurrRel + B[0:1, :]

    return preds_sc_tr_al


def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint

if __name__ == '__main__':
    a = np.random.rand(21, 3)
    b = np.random.rand(21, 3)
    a2 = align_sc_tr(a, b)
    print(a2.shape)
