import torch
import cv2
import numpy as np


def perspective2(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:, :2, :2]
        shift = transforms[:, :2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz


def perspective(points, calibrations):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :return: points_img: [Bx2xN] Tensor of uvz coordinates in the image plane
    '''
    if points.shape[1] == 2:
        points = torch.cat([points, torch.ones([points.shape[0], 1, points.shape[2]]).to(points.device)], 1)
    z = points[:, 2:3].clone()
    points[:, :3] = points[:, :3] / z
    points1 = torch.cat([points, torch.ones([points.shape[0], 1, points.shape[2]]).to(points.device)], 1)
    points_img = torch.bmm(calibrations, points1)
    points_img = torch.cat([points_img[:, :2], z], 1)

    return points_img


def perspective_np(points, calibrations):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [BxNx3] Tensor of 3D points
    :param calibrations: [Bx4x4] Tensor of projection matrix
    :return: points_img: [BxNx3] Tensor of uvz coordinates in the image plane
    '''
    if points.shape[1] == 2:
        points = np.concatenate([points, np.ones([points.shape[0], 1])], -1)
    z = points[:, 2:3].copy()
    points[:, :3] /= z
    points1 = np.concatenate([points, np.ones([points.shape[0], 1])], -1)
    points_img = np.dot(calibrations, points1.T).T
    points_img = np.concatenate([points_img[:, :2], z], -1)

    return points_img


def compute_iou(pred, gt):
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
    area = cv2.contourArea(cnt)
    return area