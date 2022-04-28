# ------------------------------------------------------------------------------
# Copyright (c) 2021
# Licensed under the MIT License.
# Written by Xingyu Chen(chenxingyusean@foxmail.com)
# ------------------------------------------------------------------------------

from __future__ import unicode_literals, print_function
import numpy as np
import cv2
from cmr.datasets.FreiHAND.kinematics import mano_to_mpii
from scipy.optimize import minimize


def base_transform(img, size, mean=0.5, std=0.5):
    x = cv2.resize(img, (size, size)).astype(np.float32) / 255
    x -= mean
    x /= std
    x = x.transpose(2, 0, 1)

    return x


def inv_base_tranmsform(x, mean=0.5, std=0.5):
    x = x.transpose(1, 2, 0)
    image = (x * std + mean) * 255
    return image.astype(np.uint8)


def crop_roi(img, bbox, out_sz, padding=(0, 0, 0)):
    bbox = [float(x) for x in bbox]
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(img, mapping, (out_sz, out_sz),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=padding)
    return crop


def registration(vertex, uv, j_regressor, K, size, uv_conf=None, poly=None):
    """
    Adaptive 2D-1D registration
    :param vertex: 3D mesh xyz
    :param uv: 2D pose
    :param j_regressor: matrix for vertex -> joint
    :param K: camera parameters
    :param size: image size
    :param uv_conf: 2D pose confidence
    :param poly: contours from silhouette
    :return: camera-space vertex
    """
    t = np.array([0, 0, 0.6])
    bounds = ((None, None), (None, None), (0.3, 2))
    poly_protect = [0.06, 0.02]

    vertex2xyz = np.matmul(j_regressor, vertex)
    if vertex2xyz.shape[0] == 21:
        vertex2xyz = mano_to_mpii(vertex2xyz)
    try_poly = True
    if uv_conf is None:
        uv_conf = np.ones([uv.shape[0], 1])
    uv_select = uv_conf > 0.1
    if uv_select.sum() == 0:
        success = False
    else:
        loss = np.array([5, ])
        attempt = 5
        while loss.mean() > 2 and attempt:
            attempt -= 1
            uv = uv[uv_select.repeat(2, axis=1)].reshape(-1, 2)
            uv_conf = uv_conf[uv_select].reshape(-1, 1)
            vertex2xyz = vertex2xyz[uv_select.repeat(3, axis=1)].reshape(-1, 3)
            sol = minimize(align_uv, t, method='SLSQP', bounds=bounds, args=(uv, vertex2xyz, K))
            t = sol.x
            success = sol.success
            xyz = vertex2xyz + t
            proj = np.matmul(K, xyz.T).T
            uvz = np.concatenate((uv, np.ones([uv.shape[0], 1])), axis=1) * xyz[:, 2:]
            loss = abs((proj - uvz).sum(axis=1))
            uv_select = loss < loss.mean() + loss.std()
            if uv_select.sum() < 13:
                break
            uv_select = uv_select[:, np.newaxis]

    if poly is not None and try_poly:
        poly = find_1Dproj(poly[0]) / size
        sol = minimize(align_poly, np.array([0, 0, 0.6]), method='SLSQP', bounds=bounds, args=(poly, vertex, K, size))
        if sol.success:
            t2 = sol.x
            d = distance(t, t2)
            if d > poly_protect[0]:
                t = t2
            elif d > poly_protect[1]:
                t = t * (1 - (d - poly_protect[1]) / (poly_protect[0] - poly_protect[1])) + t2 * ((d - poly_protect[1]) / (poly_protect[0] - poly_protect[1]))

    return vertex + t, success


def distance(x, y):
    return np.sqrt(((x - y)**2).sum())


def find_1Dproj(points):
    angles = [(0, 90), (-15, 75), (-30, 60), (-45, 45), (-60, 30), (-75, 15)]
    axs = [(np.array([[np.cos(x/180*np.pi), np.sin(x/180*np.pi)]]), np.array([np.cos(y/180*np.pi), np.sin(y/180*np.pi)])) for x, y in angles]
    proj = []
    for ax in axs:
        x = (points * ax[0]).sum(axis=1)
        y = (points * ax[1]).sum(axis=1)
        proj.append([x.min(), x.max(), y.min(), y.max()])

    return np.array(proj)


def align_poly(t, poly, vertex, K, size):
    proj = np.matmul(K, (vertex + t).T).T
    proj = (proj / proj[:, 2:])[:, :2]
    proj = find_1Dproj(proj) / size
    loss = (proj - poly)**2

    return loss.mean()


def align_uv(t, uv, vertex2xyz, K):
    xyz = vertex2xyz + t
    proj = np.matmul(K, xyz.T).T
    uvz = np.concatenate((uv, np.ones([uv.shape[0], 1])), axis=1) * xyz[:, 2:]
    loss = (proj - uvz)**2

    return loss.mean()


def map2uv(map, size=(224, 224)):
    if map.ndim == 4:
        uv = np.zeros((map.shape[0], map.shape[1], 2))
        uv_conf = np.zeros((map.shape[0], map.shape[1], 1))
        map_size = map.shape[2:]
        for j in range(map.shape[0]):
            for i in range(map.shape[1]):
                uv_conf[j][i] = map[j, i].max()
                max_pos = map[j, i].argmax()
                uv[j][i][1] = (max_pos // map_size[1]) / map_size[0] * size[0]
                uv[j][i][0] = (max_pos % map_size[1]) / map_size[1] * size[1]
    else:
        uv = np.zeros((map.shape[0], 2))
        uv_conf = np.zeros((map.shape[0], 1))
        map_size = map.shape[1:]
        for i in range(map.shape[0]):
            uv_conf[i] = map[i].max()
            max_pos = map[i].argmax()
            uv[i][1] = (max_pos // map_size[1]) / map_size[0] * size[0]
            uv[i][0] = (max_pos % map_size[1]) / map_size[1] * size[1]

    return uv, uv_conf


def uv2map(uv, size=(224, 224)):
    kernel_size = (size[0] * 13 // size[0] - 1) // 2
    gaussian_map = np.zeros((uv.shape[0], size[0], size[1]))
    size_transpose = np.array(size)
    gaussian_kernel = cv2.getGaussianKernel(2 * kernel_size + 1, (2 * kernel_size + 2)/4.)
    gaussian_kernel = np.dot(gaussian_kernel, gaussian_kernel.T)
    gaussian_kernel = gaussian_kernel/gaussian_kernel.max()

    for i in range(gaussian_map.shape[0]):
        if (uv[i] >= 0).prod() == 1 and (uv[i][1] <= size_transpose[0]) and (uv[i][0] <= size_transpose[1]):
            s_pt = np.array((uv[i][1], uv[i][0]))
            p_start = s_pt - kernel_size
            p_end = s_pt + kernel_size
            p_start_fix = (p_start >= 0) * p_start + (p_start < 0) * 0
            k_start_fix = (p_start >= 0) * 0 + (p_start < 0) * (-p_start)
            p_end_fix = (p_end <= (size_transpose - 1)) * p_end + (p_end > (size_transpose - 1)) * (size_transpose - 1)
            k_end_fix = (p_end <= (size_transpose - 1)) * kernel_size * 2 + (p_end > (size_transpose - 1)) * (2*kernel_size - (p_end - (size_transpose - 1)))
            gaussian_map[i, p_start_fix[0]: p_end_fix[0] + 1, p_start_fix[1]: p_end_fix[1] + 1] = \
                gaussian_kernel[k_start_fix[0]: k_end_fix[0] + 1, k_start_fix[1]: k_end_fix[1] + 1]

    return gaussian_map


def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area


def tensor2array(tensor, max_value=None, colormap='jet', channel_first=True, mean=0.5, std=0.5):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            color_cvt = cv2.COLOR_BGR2RGB
            if colormap == 'jet':
                colormap = cv2.COLORMAP_JET
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255 * tensor.squeeze().numpy() / max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy() / max_value).clip(0, 1)
        if channel_first:
            array = array.transpose(2, 0, 1)
    elif tensor.ndimension() == 3:
        assert (tensor.size(0) == 3)
        array = ((mean + tensor.numpy() * std) * 255).astype(np.uint8)
        if not channel_first:
            array = array.transpose(1, 2, 0)

    return array