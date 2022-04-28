import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn.functional as F
import torch


def l1_loss(pred, gt, is_valid=None, drop_nan=False):
    if drop_nan:
        pred = torch.where(torch.isnan(pred), torch.full_like(pred, 0), pred)
        pred = torch.where(torch.isinf(pred), torch.full_like(pred, 0), pred)
        gt = torch.where(torch.isnan(gt), torch.full_like(gt, 0), gt)
        gt = torch.where(torch.isinf(gt), torch.full_like(gt, 0), gt)
    loss = F.l1_loss(pred, gt, reduction='none')
    if is_valid is not None:
        loss *= is_valid
        pos_num = (loss > 0).sum()
        if pos_num == 0:
            return 0
        else:
            return loss.sum() / pos_num
    else:
        return loss.mean()


def bce_loss(pred, gt, is_valid=None):
    loss = F.binary_cross_entropy(pred, gt, reduction='none')
    if is_valid is not None:
        loss *= is_valid
        pos_num = (loss > 0).sum()
        if pos_num == 0:
            return 0
        else:
            return loss.sum() / pos_num
    else:
        return loss.mean()

# def bce_loss(pred, gt, is_valid=None):
#     loss = F.binary_cross_entropy(pred, gt, reduction='none')
#     if is_valid is not None:
#         loss *= is_valid
# valid_idx = [i for i in range(len(is_valid)) if bool(is_valid[i])]
# loss = loss[valid_idx]

# return loss.mean()


def bce_wlog_loss(pred, gt, is_valid=None):
    loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
    if is_valid is not None:
        loss *= is_valid
        pos_num = (loss > 0).sum()
        if pos_num == 0:
            return 0
        else:
            return loss.sum() / pos_num
    else:
        return loss.mean()


def edge_loss(edge, pred, gt):
    pred_start = pred.index_select(1, edge[0])
    pred_end = pred.index_select(1, edge[1])
    gt_start = gt.index_select(1, edge[0])
    gt_end = gt.index_select(1, edge[1])

    pred_length = torch.sqrt(torch.sum((pred_start - pred_end) ** 2, dim=2))
    gt_length = torch.sqrt(torch.sum((gt_start - gt_end) ** 2, dim=2))
    return l1_loss(pred_length, gt_length)


def mask_loss(pred, gt, face, K, render):
    mask = render(pred, face, K=K, mode='silhouettes')

    return F.l1_loss(mask, gt), mask


def proj_loss(pred, gt, K, img_size=224):
    uv_pred = torch.bmm(K, pred.transpose(1, 2)).transpose(1, 2)
    uv_pred = (uv_pred / uv_pred[:, :, 2:3])[:, :, :2] / img_size
    uv_gt = torch.bmm(K, gt.transpose(1, 2)).transpose(1, 2)
    uv_gt = (uv_gt / uv_gt[:, :, 2:3])[:, :, :2] / img_size

    return F.l1_loss(uv_pred.clamp(min=0., max=1.), uv_gt.clamp(min=0., max=1.))


def normal_loss(pred, gt, face, is_valid=None):

    v1_out = pred[:, face[:, 1], :] - pred[:, face[:, 0], :]
    v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
    v2_out = pred[:, face[:, 2], :] - pred[:, face[:, 0], :]
    v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
    v3_out = pred[:, face[:, 2], :] - pred[:, face[:, 1], :]
    v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

    v1_gt = gt[:, face[:, 1], :] - gt[:, face[:, 0], :]
    v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
    v2_gt = gt[:, face[:, 2], :] - gt[:, face[:, 0], :]
    v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
    normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
    normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

    # valid_mask = valid[:, face[:, 0], :] * valid[:, face[:, 1], :] * valid[:, face[:, 2], :]

    cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True)) #* valid_mask
    cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True)) #* valid_mask
    cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True)) #* valid_mask
    loss = torch.cat((cos1, cos2, cos3), 1)
    if is_valid is not None:
        # valid_idx = [i for i in range(len(is_valid)) if bool(is_valid[i])]
        # loss = loss[valid_idx]
        loss *= is_valid
    return loss.mean()


def edge_length_loss(pred, gt, face, is_valid=None):

    d1_out = torch.sqrt(torch.sum((pred[:, face[:, 0], :] - pred[:, face[:, 1], :]) ** 2, 2, keepdim=True))
    d2_out = torch.sqrt(torch.sum((pred[:, face[:, 0], :] - pred[:, face[:, 2], :]) ** 2, 2, keepdim=True))
    d3_out = torch.sqrt(torch.sum((pred[:, face[:, 1], :] - pred[:, face[:, 2], :]) ** 2, 2, keepdim=True))

    d1_gt = torch.sqrt(torch.sum((gt[:, face[:, 0], :] - gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
    d2_gt = torch.sqrt(torch.sum((gt[:, face[:, 0], :] - gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
    d3_gt = torch.sqrt(torch.sum((gt[:, face[:, 1], :] - gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))

    # valid_mask_1 = valid[:, face[:, 0], :] * valid[:, face[:, 1], :]
    # valid_mask_2 = valid[:, face[:, 0], :] * valid[:, face[:, 2], :]
    # valid_mask_3 = valid[:, face[:, 1], :] * valid[:, face[:, 2], :]

    diff1 = torch.abs(d1_out - d1_gt) #* valid_mask_1
    diff2 = torch.abs(d2_out - d2_gt) #* valid_mask_2
    diff3 = torch.abs(d3_out - d3_gt) #* valid_mask_3
    loss = torch.cat((diff1, diff2, diff3), 1)
    if is_valid is not None:
        # valid_idx = [i for i in range(len(is_valid)) if bool(is_valid[i])]
        # loss = loss[valid_idx]
        loss *= is_valid
    return loss.mean()


def compute_iou(pred, gt):
    # import cv2
    # cv2.imshow('test', gt*255)
    # cv2.waitKey(0)
    area_pred = pred.sum()
    area_gt = gt.sum()
    if area_pred == area_gt == 0:
        return 1
    union_area = (pred + gt).clip(max=1)
    union_area = union_area.sum()
    inter_area = area_pred + area_gt - union_area
    IoU = inter_area / union_area

    return IoU


def contrastive_loss_3d(verts, aug_param):
    rot_mat = torch.empty(aug_param.size()[0], 3, 3)
    rot_angle = aug_param[:, 4] - aug_param[:, 0]
    ang_rad = torch.deg2rad(rot_angle)
    for i in range(aug_param.size()[0]):
        rot_mat[i] = torch.tensor([[torch.cos(ang_rad[i]), torch.sin(ang_rad[i]), 0],
                                   [-torch.sin(ang_rad[i]), torch.cos(ang_rad[i]), 0],
                                   [0, 0, 1]])
    verts_rot = torch.bmm(rot_mat.to(verts.device), verts[..., :3].permute(0, 2, 1)).permute(0, 2, 1)

    return F.l1_loss(verts_rot, verts[..., 3:], reduction='mean')


def contrastive_loss_2d(uv_pred, uv_trans, size):
    #取出UV坐标
    uv_pred_pre = uv_pred[:, :, :2]
    uv_pred_lat = uv_pred[:, :, 2:]

    uv_trans_pre = uv_trans[:, :, :3]
    uv_trans_lat = uv_trans[:, :, 3:]

    # 计算uv reverse坐标
    uv_pred_pre_rev = revtrans_points(uv_pred_pre * size, uv_trans_pre) / size
    uv_pred_lat_rev = revtrans_points(uv_pred_lat * size, uv_trans_lat) / size

    # loss = torch.sqrt(torch.sum(torch.pow(uv_pred_pre_rev - uv_pred_lat_rev, 2), 2))
    return F.l1_loss(uv_pred_pre_rev, uv_pred_lat_rev, reduction='mean')


def revtrans_points(uv_point, trans):
    uv1 = torch.cat((uv_point, torch.ones_like(uv_point[:, :, :1])), 2)
    uv_crop = torch.bmm(trans, uv1.transpose(2, 1)).transpose(2, 1)[:, :, :2]

    return uv_crop
