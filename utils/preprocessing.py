import numpy as np
import cv2
import random
import math
from utils.augmentation import get_m1to1_gaussian_rand


def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == 'RGB':
        img = img[:, :, ::-1].copy()

    img = img
    return img


def get_bbox(joint_img, joint_valid):
    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1];
    y_img = y_img[joint_valid == 1];
    xmin = min(x_img);
    ymin = min(y_img);
    xmax = max(x_img);
    ymax = max(y_img);

    x_center = (xmin + xmax) / 2.;
    width = xmax - xmin;
    xmin = x_center - 0.5 * width * 1.2
    xmax = x_center + 0.5 * width * 1.2

    y_center = (ymin + ymax) / 2.;
    height = ymax - ymin;
    ymin = y_center - 0.5 * height * 1.2
    ymax = y_center + 0.5 * height * 1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def process_bbox(bbox, img_width, img_height, input_img_shape=(256, 256)):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = input_img_shape[1] / input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    return bbox


def get_aug_config(exclude_flip, base_scale=1.1, scale_factor=0.25, rot_factor=60, color_factor=0.2, gaussian_std=1):
    # scale_factor = 0.25
    # rot_factor = 60
    # color_factor = 0.2

    # scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    scale = get_m1to1_gaussian_rand(gaussian_std) * scale_factor + base_scale
    # rot = np.clip(np.random.randn(), -2.0, 2.0) * rot_factor if random.random() <= 0.6 else 0
    rot = get_m1to1_gaussian_rand(gaussian_std) * rot_factor if random.random() <= 0.6 else 0
    shift = [get_m1to1_gaussian_rand(gaussian_std), get_m1to1_gaussian_rand(gaussian_std)]

    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    if exclude_flip:
        do_flip = False
    else:
        do_flip = random.random() <= 0.5

    return scale, rot, shift, color_scale, do_flip


def augmentation(img, bbox, data_split, exclude_flip=False, input_img_shape=(256, 256), mask=None, base_scale=1.1, scale_factor=0.25, rot_factor=60, shift_wh=None, gaussian_std=1, color_aug=False):
    if data_split == 'train':
        scale, rot, shift, color_scale, do_flip = get_aug_config(exclude_flip, base_scale=base_scale, scale_factor=scale_factor, rot_factor=rot_factor, gaussian_std=gaussian_std)
    else:
        scale, rot, shift, color_scale, do_flip = base_scale, 0.0, [0, 0], np.array([1, 1, 1]), False

    img, trans, inv_trans, mask, shift_xy = generate_patch_image(img, bbox, scale, rot, shift, do_flip, input_img_shape, shift_wh=shift_wh, mask=mask)
    if color_aug:
        img = np.clip(img * color_scale[None, None, :], 0, 255)
    return img, trans, inv_trans, np.array([rot, scale, *shift_xy]), do_flip, input_img_shape[0]/(bbox[3]*scale), mask


def augmentation_2d(img, joint_img, princpt, trans, do_flip):
    joint_img = joint_img.copy()
    joint_num = len(joint_img)
    original_img_shape = img.shape

    if do_flip:
        joint_img[:, 0] = original_img_shape[1] - joint_img[:, 0] - 1
        princpt[0] = original_img_shape[1] - princpt[0] - 1
    for i in range(joint_num):
        joint_img[i,:2] = trans_point2d(joint_img[i,:2], trans)
    princpt = trans_point2d(princpt, trans)
    return joint_img, princpt


def generate_patch_image(cvimg, bbox, scale, rot, shift, do_flip, out_shape, shift_wh=None, mask=None):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1
        if mask is not None:
            mask = mask[:, ::-1]

    trans, shift_xy = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, shift, shift_wh=shift_wh, return_shift=True)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    if mask is not None:
        mask = cv2.warpAffine(mask, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
        mask = (mask > 150).astype(np.uint8)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, shift, shift_wh=shift_wh, inv=True)

    return img_patch, trans, inv_trans, mask, shift_xy


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, shift, shift_wh=None, inv=False, return_shift=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    if shift_wh is not None:
        shift_lim = (max((src_w - shift_wh[0]) / 2, 0), max((src_h - shift_wh[1]) / 2, 0))
        x_shift = shift[0] * shift_lim[0]
        y_shift = shift[1] * shift_lim[1]
    else:
        x_shift = y_shift = 0
    src_center = np.array([c_x + x_shift, c_y + y_shift], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    if return_shift:
        return trans, [x_shift/src_w, y_shift/src_h]
    return trans


def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]