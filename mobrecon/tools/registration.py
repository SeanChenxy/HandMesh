# Copyright (c) Xingyu Chen. All Rights Reserved.

"""
 * @file registration.py
 * @author chenxingyu (chenxy.sean@gmail.com)
 * @brief mesh registration with adaptive 2D-1D method
 * @version 0.1
 * @date 2022-04-28
 * 
 * @copyright Copyright (c) 2022 chenxingyu
 * 
"""

import numpy as np
from mobrecon.tools.kinematics import mano_to_mpii
from mobrecon.tools.vis import perspective_np
from scipy.optimize import minimize


def registration(vertex, uv, j_regressor, calib, size, uv_conf=None, poly=None):
    """Adaptive 2D-1D registration

    Args:
        vertex (array): 3D vertex coordinates in hand frame
        uv (array): 2D landmarks
        j_regressor (array): vertex -> joint
        calib (array): intrinsic camera parameters
        size (int): image shape
        uv_conf (array, optional): confidence of 2D landmarks. Defaults to None.
        poly (array, optional): _description_. Defaults to None.

    Returns:
        array: camera-space vertex
    """
    t = np.array([0, 0, 0.6])
    bounds = ((None, None), (None, None), (0.05, 2))
    poly_protect = [0.06, 0.02]

    vertex2xyz = mano_to_mpii(np.matmul(j_regressor, vertex))
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
            sol = minimize(align_uv, t, method='SLSQP', bounds=bounds, args=(uv, vertex2xyz, calib))
            t = sol.x
            success = sol.success
            xyz = vertex2xyz + t
            proj = perspective_np(xyz, calib)[:, :2]
            loss = abs((proj - uv).sum(axis=1))
            uv_select = loss < loss.mean() + loss.std()
            if uv_select.sum() < 13:
                break
            uv_select = uv_select[:, np.newaxis]

    if poly is not None and try_poly:
        poly = find_1Dproj(poly[0]) / size
        sol = minimize(align_poly, np.array([0, 0, 0.6]), method='SLSQP', bounds=bounds, args=(poly, vertex, calib, size))
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


def align_poly(t, poly, vertex, calib, size):
    proj = perspective_np((vertex + t), calib)[:, :2]
    proj = find_1Dproj(proj) / size
    loss = (proj - poly)**2

    return loss.mean()


def align_uv(t, uv, vertex2xyz, calib):
    xyz = vertex2xyz + t
    proj = perspective_np(xyz, calib)[:, :2]
    loss = (proj - uv)**2

    return loss.mean()
