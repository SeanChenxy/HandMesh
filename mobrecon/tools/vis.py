# Copyright (c) Xingyu Chen. All Rights Reserved.

"""
 * @file vis.py
 * @author chenxingyu (chenxy.sean@gmail.com)
 * @brief some visual computation
 * @version 0.1
 * @date 2022-04-28
 * 
 * @copyright Copyright (c) 2022 chenxingyu
 * 
"""

import torch
import cv2
import numpy as np


def perspective(points, calibrations):
    """Compute the perspective projections of 3D points into the image plane by given projection matrix

    Args:
        points (tensot): [Bx3xN] tensor of 3D points
        calibrations (tensor): [Bx4x4] Tensor of projection matrix

    Returns:
        tensor: [Bx3xN] Tensor of uvz coordinates in the image plane
    """
    if points.shape[1] == 2:
        points = torch.cat([points, torch.ones([points.shape[0], 1, points.shape[2]]).to(points.device)], 1)
    z = points[:, 2:3].clone()
    points[:, :3] = points[:, :3] / z
    points1 = torch.cat([points, torch.ones([points.shape[0], 1, points.shape[2]]).to(points.device)], 1)
    points_img = torch.bmm(calibrations, points1)
    points_img = torch.cat([points_img[:, :2], z], 1)

    return points_img


def perspective_np(points, calibrations):
    """Compute the perspective projections of 3D points into the image plane by given projection matrix

    Args:
        points (array): [BxNx3] array of 3D points
        calibrations (array): [Bx4x4] Tensor of projection matrix

    Returns:
        array: [BxNx3] Tensor of uvz coordinates in the image plane
    """
    if points.shape[1] == 2:
        points = np.concatenate([points, np.ones([points.shape[0], 1])], -1)
    z = points[:, 2:3].copy()
    points[:, :3] /= z
    points1 = np.concatenate([points, np.ones([points.shape[0], 1])], -1)
    points_img = np.dot(calibrations, points1.T).T
    points_img = np.concatenate([points_img[:, :2], z], -1)

    return points_img

def compute_iou(pred, gt):
    """Mask IoU

    Args:
        pred (array): prediction mask
        gt (array): ground-truth mask

    Returns:
        float: IoU
    """
    area_pred = pred.sum()
    area_gt = gt.sum()
    if area_pred == area_gt == 0:
        return 1
    union_area = (pred + gt).clip(max=1)
    union_area = union_area.sum()
    inter_area = area_pred + area_gt - union_area
    IoU = inter_area / union_area

    return IoU


def cnt_area(cnt):
    """Compute area of a contour

    Args:
        cnt (array): contour

    Returns:
        float: area
    """
    area = cv2.contourArea(cnt)
    return area