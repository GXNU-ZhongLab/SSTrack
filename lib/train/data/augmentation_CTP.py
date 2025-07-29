# Copyright (c) SenseTime. All Rights Reserved.
import numpy as np
import cv2
import random
import torchvision.transforms as transforms
from lib.utils.box_ops import corner2center, Center, center2corner, Corner
from PIL import ImageFilter

import lib.train.data.transforms_mt as tfm  # this file has been modified from the transforms.py in the project of pytracking
import torch


def random_masking(img):
    R = np.mean(img[:, :, 0])
    G = np.mean(img[:, :, 1])
    B = np.mean(img[:, :, 2])
    R, G, B = np.uint8(R), np.uint8(G), np.uint8(B)
    avg = np.mean(img)

    mask = np.zeros(img.shape)
    mask_matrix = np.random.randint(1, 10, [32, 32, 1])
    # r = np.random.randint(3, 7, 1)
    r = 6
    mask_matrix[mask_matrix >= r] = 1
    mask_matrix[mask_matrix != 1] = 0
    mask += mask_matrix
    # print(mask_matrix)
    img = img * np.uint8(mask)
    img[img == 0] = avg

    return img
    # print(R, G, B)


def mixup(dataset, origin_img, origin_labels, input_dim, mixup_scale=(0.5, 1.5)):
    jit_factor = random.uniform(*mixup_scale)
    FLIP = random.uniform(0, 1) > 0.5
    cp_labels = []
    while len(cp_labels) == 0:
        cp_index = random.randint(0, len(dataset) - 1)
        cp_labels = [dataset[cp_index][1]]
    img, cp_labels = np.array(dataset[cp_index][0]), [dataset[cp_index][1]]

    if len(img.shape) == 3:
        cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
    else:
        cp_img = np.ones(input_dim, dtype=np.uint8) * 114

    cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
        interpolation=cv2.INTER_LINEAR,
    )

    cp_img[
    : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
    ] = resized_img
    cp_img = cv2.resize(
        cp_img,
        (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
    )
    cp_scale_ratio *= jit_factor

    if FLIP:
        cp_img = cp_img[:, ::-1, :]
    origin_h, origin_w = cp_img.shape[:2]
    target_h, target_w = origin_img.shape[:2]
    padded_img = np.zeros(
        (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
    )
    padded_img[:origin_h, :origin_w] = cp_img
    origin_img = origin_img.astype(np.float32)

    x_offset, y_offset = 0, 0
    if padded_img.shape[0] > target_h:
        y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
    if padded_img.shape[1] > target_w:
        x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
    padded_cropped_img = padded_img[
                         y_offset: y_offset + target_h, x_offset: x_offset + target_w
                         ]

    origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)
    return origin_img.astype(np.uint8), origin_labels


def random_perspective(img, targets, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    '''
    遍性数据增强：
            进行随机旋转，缩放，错切，平移，center，perspective数据增强
    Args:
        img: shape=(height, width, 3)
        targets ：size = (单张图片中的目标数量, [class, xyxy, Θ])
    Returns:
        img：shape=(height, width, 3)
        targets = (目标数量, [cls, xyxy, Θ])
    '''

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # 设置旋转和缩放的仿射矩阵并进行旋转和缩放
    # Rotation and Scale
    R = np.eye(3)  # 行数为3,对角线为1,其余为0的矩阵
    a = random.uniform(-degrees, degrees)   # 随机生成[-degrees, degrees)的实数 即为旋转角度 负数则代表逆时针旋转
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)  # 获得以(0,0)为中心的旋转仿射变化矩阵

    # 设置裁剪的仿射矩阵系数
    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # 设置平移的仿射系数
    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    # 融合仿射矩阵并作用在图片上
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    return img, targets


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


#  ===========================


def random_mask(mask_sz, edge_lens=None):
   # this function generates a random mask whose external matrix is the given patch box.
   # mask_sz - [patch_height, patch_width]
   # edge_lens - (lengths of top, bottom, left, and right edges)
    ph, pw = mask_sz
    if edge_lens is None:
        min_width, min_height = 4, 4  #minimum length of the intersection line of the mask and the box edge.
        edge_len_ratio=0.6
        v_line_left_len, v_line_right_len = min(ph,max(round(np.random.random()*edge_len_ratio*ph),min_height)), min(ph,max(round(np.random.random()*edge_len_ratio*ph),min_height))  # top edge length of mask
        h_line_top_len, h_line_bottom_len = max(round(np.random.random()*edge_len_ratio * pw), min_width), max(round(np.random.random()*edge_len_ratio * pw), min_width)  # left edge length of mask
        mask_num = 1
    else:
        h_line_top_len, h_line_bottom_len,v_line_left_len, v_line_right_len = edge_lens
        mask_num = 6
    #initialize the mask with 0s
    mask = np.zeros((ph,pw,mask_num))
    for mask_i in range(mask_num):
        top_s, bottom_s = random.randint(0, pw - h_line_top_len), random.randint(0, pw - h_line_bottom_len)
        left_s, right_s = random.randint(0, ph - v_line_left_len), random.randint(0, ph - v_line_right_len)
        #compute the start and end of each row
        left_starts1 = list(np.random.randint(0,top_s+1, left_s))
        left_starts1.sort(reverse = True)
        left_starts2 = list(np.random.randint(0,bottom_s+1, ph-left_s-v_line_left_len))
        left_starts2.sort()
        left_starts = left_starts1 + [0]*v_line_left_len + left_starts2

        right_starts1 = list(np.random.randint(top_s+h_line_top_len,pw+1,right_s))
        right_starts1.sort()
        right_starts2 = list(np.random.randint(bottom_s+h_line_bottom_len, pw+1, ph-v_line_right_len-right_s))
        right_starts2.sort(reverse = True)
        right_ends = right_starts1 + [pw]*v_line_right_len + right_starts2

        #and then fill it with 1s, and fill it row by row
        for row_i in range(ph):
            mask[row_i,left_starts[row_i]:right_ends[row_i],mask_i] = 1

    return mask


def random_masks(mask_sz):
   # this function generates a random mask whose external matrix is the given patch box.
   # mask_sz - [patch_height, patch_width]
   # edge_lens - (lengths of top, bottom, left, and right edges)
    ph, pw = mask_sz

    min_width, min_height = 4, 4  #minimum length of the intersection line of the mask and the box edge.
    edge_len_ratio=0.6
    v_line_left_len, v_line_right_len = min(ph,max(round(np.random.random()*edge_len_ratio*ph),min_height)), min(ph,max(round(np.random.random()*edge_len_ratio*ph),min_height))  # top edge length of mask
    h_line_top_len, h_line_bottom_len = max(round(np.random.random()*edge_len_ratio * pw), min_width), max(round(np.random.random()*edge_len_ratio * pw), min_width)  # left edge length of mask


    top_s, bottom_s = random.randint(0, pw - h_line_top_len), random.randint(0, pw - h_line_bottom_len)
    left_s, right_s = random.randint(0, ph - v_line_left_len), random.randint(0, ph - v_line_right_len)

    #compute the start and end of each row
    left_starts1 = list(np.random.randint(0,top_s+1, left_s))
    left_starts1.sort(reverse = True)
    left_starts2 = list(np.random.randint(0,bottom_s+1, ph-left_s-v_line_left_len))
    left_starts2.sort()
    left_starts = left_starts1 + [0]*v_line_left_len + left_starts2

    right_starts1 = list(np.random.randint(top_s+h_line_top_len,pw+1,right_s))
    right_starts1.sort()
    right_starts2 = list(np.random.randint(bottom_s+h_line_bottom_len, pw+1, ph-v_line_right_len-right_s))
    right_starts2.sort(reverse = True)
    right_ends = right_starts1 + [pw]*v_line_right_len + right_starts2

    #initialize the mask with 0s
    mask = np.zeros((ph,pw))
    #and then fill it with 1s, and fill it row by row
    for row_i in range(ph):
        mask[row_i,left_starts[row_i]:right_ends[row_i]] = 1

    return mask


class PasteObjects(object):
    def __init__(self, max_bgo_num=5, mem_size=20):
        # bgo_img -  a random image patch from the background for initialization
        self.max_bgo_num = max_bgo_num  # maximum of synthesized background objects in one sample
        self.mem_size = mem_size
        self.mem = []

    def update_mem(self, obj_img):
        # since the list is shuffled before update, we directly pop the first item
        if len(self.mem) == self.mem_size:
            self.mem.pop(0)
            # del a
        self.mem.append(obj_img)

    def get_mem_size(self):
        return len(self.mem)

    def paste_objects(self, image, num, paste_scope=None):  # return a list of selected objects
        # paste the selected objects onto the base image
        # bbox - bounding box of the target
        num = min(len(self.mem), num)
        H, W, _ = image.shape
        if paste_scope is None:
            paste_pad_x, paste_pad_y = 5, 5
        else:
            paste_pad_x, paste_pad_y = (W - paste_scope) // 2, (H - paste_scope) // 2

        def gen_mask(p_sz, pad_sz=[5, 5]):  # p_sz: size of the patch
            pw, ph = p_sz
            pad_sz[0], pad_sz[1] = min(pad_sz[0], pw // 2), min(pad_sz[1], ph // 2)
            h_inds = np.zeros(ph, dtype=np.float32)
            w_inds = np.zeros(pw, dtype=np.float32)
            w_inds[:pad_sz[0]] = np.arange(pad_sz[0], 0, -1)
            w_inds[-pad_sz[0]:] = np.arange(1, pad_sz[0] + 1)
            h_inds[:pad_sz[1]] = np.arange(pad_sz[1], 0, -1)
            h_inds[-pad_sz[1]:] = np.arange(1, pad_sz[1] + 1)

            h_inds = np.tile(h_inds.reshape(-1, 1) * 1.0 / pad_sz[1], (1, pw))
            w_inds = np.tile((w_inds * 1.0 / pad_sz[0]), (ph, 1))
            # inds,_ = torch.max(torch.cat((h_inds.reshape(1,-1),w_inds.reshape(1,-1)),0),0)
            inds = (h_inds + w_inds)
            inds = np.where(inds > 1, np.ones_like(inds), inds)
            return inds

        p_ratio = random.choices([4, 5, 6, 7], k=2)
        for obj_i in range(num):
            obj_h, obj_w = min(self.mem[obj_i].shape[0], H // p_ratio[0]), min(self.mem[obj_i].shape[1], W // p_ratio[1])
            # First, get the random locations to paste at
            xs = random.choice(range(paste_pad_x, W - obj_w - paste_pad_x))
            ys = random.choice(range(paste_pad_y, H - obj_h - paste_pad_y))
            # Second, perform the pasting
            mask = np.expand_dims(gen_mask((obj_w, obj_h), [1, 1]), 2)
            image[ys:ys + obj_h, xs:xs + obj_w, :] = mask * image[ys:ys + obj_h, xs:xs + obj_w, :] \
                                                     + (1 - mask) * self.mem[obj_i][:obj_h, :obj_w, :]
        return image

    def __call__(self, image):
        # image - the base image, bbox - target box with the Corner format [x1,y1,x2,y2]
        random.shuffle(self.mem)
        bgo_num = random.choice(np.arange(1, self.max_bgo_num + 1))
        image_syn = self.paste_objects(image, bgo_num)  # ,360)
        return image_syn
    

class Augmentation_CTP:
    def __init__(self, shift, scale, blur, flip, color, occ=0, shear=0, crop=0, colorj=0, affine=0, IS=0):
        self.shift = shift
        self.scale = scale
        self.flip = flip
        self.color = color
        self.rgbVar = np.array(
            [[-0.55919361, 0.98062831, - 0.41940627],
             [1.72091413, 0.19879334, - 1.82968581],
             [4.64467907, 4.73710203, 4.88324118]], dtype=np.float32)
        self.blur_occ_IS = blur + occ + IS
        if self.blur_occ_IS > 0:
            self.blur = blur / self.blur_occ_IS
            self.occ = occ / self.blur_occ_IS +self.blur
            self.IS = IS / self.blur_occ_IS
        else:
            self.blur = blur
            self.occ = occ
            self.IS = IS
        self.colorj = colorj
        #print('color jitter {}'.format(str(colorj)))
        self.color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # p = 0.8

        self.crop = crop
        self.crop_ratio = 0.7 # max crop ratio
        self.shear = shear
        self.affine = affine
        self.toshear = tfm.Transform(tfm.RandomAffine(max_rotation=0,max_shear=0.2,
                                                max_scale=0,max_ar_factor=0.0),)   # ar 0.3-(0.74,1.34) 0.5-(0.6,1.65) 0.2(0.82,1.22) 0.1(0.9 1.1)
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range
        #[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # input: Tensor image of size (C, H, W) to be normalized
                             std=[0.229, 0.224, 0.225])                    # outputA: Normalized Tensor image.

    @staticmethod
    def random():
        return np.random.random() * 2 - 1.0

    def _crop_roi(self, image, bbox, out_sz, padding=(0, 0, 0)):
        bbox = [float(x) for x in bbox]
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _crop_roi_new(self, image, bbox, out_sz, padding=(0, 0, 0)):
        # out_sz can be a 2-item tuple, such as out_sz = (w,h)
        if 1 == len(out_sz):
            out_sz = (out_sz, out_sz)
        bbox = [float(x) for x in bbox]
        a = (out_sz[0] - 1) / (bbox[2] - bbox[0])
        b = (out_sz[1] - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)

        crop = cv2.warpAffine(image, mapping, (int(out_sz[0]), int(out_sz[1])),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)

        return crop

    def _blur_aug(self, image):
        def rand_kernel():
            sizes = np.arange(5, 46, 2)  # (5, 46, 2)
            size = np.random.choice(sizes)
            kernel = np.zeros((size, size))
            c = int(size / 2)
            wx = np.random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1 - wx)
            return kernel, [size,wx]

        kernel, blur_pa = rand_kernel()
        image = cv2.filter2D(image, -1, kernel)
        return image, blur_pa

    def _blur_moco_aug(self, image):
        if 0.3 > np.random.random():
            return self._blur_aug(image)
        sizes = np.arange(3, 13, 2)
        sigma = np.random.choice(sizes)
        return cv2.GaussianBlur(image, (sigma, sigma), 0)

    def _color_aug(self, image):
        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset
        # print(offset)
        return image

    def _gray_aug(self, image):
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
        return image

    def _shift_scale_aug(self, image, bbox, crop_bbox, size):  # crop_bbox: box of the cropped image patch with the size: 255 for search region
        im_h, im_w = image.shape[:2]

        # adjust crop bounding box
        crop_bbox_center = corner2center(crop_bbox)
        if self.scale:
            scale_x = (1.0 + Augmentation_CTP.random() * self.scale)
            scale_y = (1.0 + Augmentation_CTP.random() * self.scale)
            #print(scale_x/scale_y)
            h, w = crop_bbox_center.h, crop_bbox_center.w
            scale_x = min(scale_x, float(im_w) / w)
            scale_y = min(scale_y, float(im_h) / h)
            # print('scale_x '+str(scale_x) +' scale_y ' + str(scale_y))
            crop_bbox_center = Center(crop_bbox_center.x,
                                      crop_bbox_center.y,
                                      crop_bbox_center.w * scale_x,
                                      crop_bbox_center.h * scale_y)

        crop_bbox = center2corner(crop_bbox_center)
        if self.shift and False:
            sx = Augmentation_CTP.random() * self.shift
            sy = Augmentation_CTP.random() * self.shift

            x1, y1, x2, y2 = crop_bbox

            sx = max(-x1, min(im_w - 1 - x2, sx))
            sy = max(-y1, min(im_h - 1 - y2, sy))

            crop_bbox = Corner(x1 + sx, y1 + sy, x2 + sx, y2 + sy)

        # adjust target bounding box
        x1, y1 = crop_bbox.x1, crop_bbox.y1
        # bbox = Corner(bbox.x1 - x1, bbox.y1 - y1,
        #               bbox.x2 - x1, bbox.y2 - y1)
        bbox = Corner(bbox.x1 - x1, bbox.y1 - y1,
                      bbox.x2 - x1, bbox.y2 - y1)
        if self.scale:
            bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y,
                          bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_roi(image, crop_bbox, size)
        return image, bbox#, [sx, sy, scale_x, scale_y]

    def scale_aug(self, image, bbox, max_shape):  # crop_bbox: box of the cropped image patch with the size: 255 for search region
        #max_shape - [w,h]
        im_h, im_w = image.shape[:2]
        # adjust crop bounding box
        if self.scale:
            scale_x = (1.0 + Augmentation_CTP.random() * self.scale)
            scale_y = (1.0 + Augmentation_CTP.random() * self.scale)

            scale_x = min(scale_x, 0.8*max_shape[1]/float(im_w))
            scale_y = min(scale_y, 0.8*max_shape[0]/float(im_h))
            bbox = Corner(round(bbox.x1 * scale_x), round(bbox.y1 * scale_y),
                          round(bbox.x2 * scale_x), round(bbox.y2 * scale_y))
            image = cv2.resize(image,(round(im_w*scale_x),round(im_h*scale_y)))
        return image, bbox#, [sx, sy, scale_x, scale_y]

    def _flip_aug(self, image, bbox):
        image = cv2.flip(image, 1)
        width = image.shape[1]
        bbox = Corner(width - 1 - bbox.x2, bbox.y1,
                      width - 1 - bbox.x1, bbox.y2)
        return image, bbox

    def _crop_aug(self, image, bbox):
        # added by xin
        thres = self.crop_ratio

        W, H = image.shape[1], image.shape[0]
        x1, y1, x2, y2 = int(max(1, round(bbox.x1))), int(max(1, int(bbox.y1))), int(min(W - 1, round(bbox.x2))), int(
            min(H - 1, round(bbox.y2)))
        w0, h0 = x2 - x1 + 1, y2 - y1 + 1
        w = round((np.random.random() * (1 - thres) + thres) * w0)
        h = round((np.random.random() * (1 - thres) + thres) * h0)

        xx = round((w0 - w) * np.random.random())
        yy = round((h0 - h) * np.random.random())

        crop_bbox = Corner(x1 + xx, y1 + yy, x1 + xx + w, y1 + yy + h) # do not minus 1 with w or h, as crop_roi_new compute w as w=x2-x1
        # print('w0 and h0 are' + str(w0) + ' '+ str(h0))
        # print('x1, x2 are' + str(x1)+' ' + str(x2) )
        patch = self._crop_roi_new(image, crop_bbox, (w0, h0), padding=(0, 0, 0))

        # print(patch.shape)
        # print('x1 x2 and w0 are' + str(x1)+' ' + str(x2)+' ' +str(w0))
        image[y1:y2+1, x1:x2+1, :] = patch
        # bbox = Corner(x1 , y1 , x2 , y2)
        return image  # , bbox

    def _color_jitter_aug(self,image):
        # input
        img3 = self.to_pil(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
        img3 = self.color_jitter(img3)
        img3 = cv2.cvtColor(np.asarray(img3), cv2.COLOR_RGB2BGR)
        return img3

    # def crop_affine_aug(self, image, bbox, SZHW): # the shear operation is optimized to make the bbox tight in the
    #     #the input and output bbox are with the format of Corner (x1,y1,x2,y2)
    #     box = torch.tensor([bbox.x1, bbox.y1, bbox.x2 - bbox.x1 + 1, bbox.y2 - bbox.y1 + 1],dtype=torch.float32)
    #     img3, box = self.toshear(image=image, bbox=box)
    #     box = box.numpy()
    #     # bbox[2], bbox[3] = max(bbox[2], 20), max(bbox[3], 20)
    #
    #     target_box=Corner(box[0], box[1], box[0] + box[2]-1, box[1] + box[3]-1)
    #
    #     target, target_box = self.crop_with_pad(img3, target_box,SZHW)
    #
    #     return target, target_box

    def affine_aug(self, image, bbox): # the shear operation is optimized to make the bbox tight in the
        #the input and output bbox are with the format of Corner (x1,y1,x2,y2)
        bbox = torch.Tensor([bbox.x1, bbox.y1, bbox.x2 - bbox.x1 + 1, bbox.y2 - bbox.y1 + 1])
        img3, bbox = self.toshear(image=image, bbox=bbox)
        bbox = bbox.numpy()

        return img3, Corner(bbox[0], bbox[1], bbox[0]+bbox[2]-1, bbox[1]+bbox[3]-1)

    # def shear_aug(self, image, bbox):
    #     #the input and output bbox are with the format of Corner (x1,y1,x2,y2)
    #
    #     y1, y2, x1, x2 = int(bbox.y1), int(bbox.y2), int(bbox.x1), int(bbox.x2)
    #     bbox = torch.tensor([bbox.x1, bbox.y1, bbox.x2 - bbox.x1 + 1, bbox.y2 - bbox.y1 + 1])
    #     target_patch = image[y1:y2+1, x1:x2+1,:]
    #     hh0,ww0 = target_patch.shape[:2]
    #     target_patch, bbox_new = self.toaffine(image=target_patch,bbox=bbox)
    #     bbox_new = bbox_new.numpy()
    #     hh1, ww1 = target_patch.shape[:2]
    #     h_pad, w_pad = (hh1-hh0)//2, (ww1-ww0)//2
    #     #print('pads '+str(h_pad)+' ' +str(w_pad))
    #
    #     image[y1:y2 + 1, x1:x2 + 1, :] = target_patch[h_pad:h_pad+hh0, w_pad:w_pad+ww0,:]
    #     return image, bbox

    def occ_mask(self,image, bbox):
    #put masks (occlusion) onto the target area to simulate the case of occlusion
        h, w = bbox.y2-bbox.y1, bbox.x2 - bbox.x1
        occ_ratio = np.sqrt((0.25+0.1*np.random.random()))      # 0.25 - 0.5

        if h<12 or w<12:
            return image
        if h/w>5 or h/w<0.2:
            return image
        #different kinds of occlusions, one and several
        if np.random.random()<1.5: # one block
            occ_w, occ_h = int(occ_ratio * w), int(occ_ratio * h)
            occ_y = random.choice(np.arange(0,int(h-occ_h))) + int(bbox.y1)
            occ_x = random.choice(np.arange(0,int(w-occ_w))) + int(bbox.x1)
            image[occ_y:occ_y+occ_h,occ_x:occ_x+occ_w,: ] = random.choice(np.arange(0,10))
        else:  # four blocks 0.2
            occ_w, occ_h = int(0.25 * w), int(0.25 * h)
            occ_y = random.choice(np.arange(0,int(occ_h))) + int(bbox.y1)
            occ_x = random.choice(np.arange(0,int(occ_w))) + int(bbox.x1)
            fill_val = random.choice(np.arange(0, 10))
            image[occ_y:occ_y + occ_h, occ_x:occ_x + occ_w, :] = fill_val
            image[occ_y+occ_h*2:occ_y + occ_h*3, occ_x:occ_x + occ_w, :] = fill_val
            image[occ_y:occ_y + occ_h, occ_x+2*occ_w:occ_x + 3*occ_w, :] = fill_val
            image[occ_y+occ_h*2:occ_y + occ_h*3, occ_x+2*occ_w:occ_x + 3*occ_w, :] = fill_val
        return image

    def intra_shift(self, image, bbox):
        # perform intra-shift of the target
        # case1 only move part of the target; case2
        def circshift(patch, shiftnum_down, shiftnum_left):
            h, w = patch.shape[:2]
            patch = np.vstack((patch[(h - shiftnum_down):, :,:], patch[:(h - shiftnum_down), :,:]))
            patch = np.hstack((patch[:, (w - shiftnum_left):,:], patch[:, :(w - shiftnum_left),:]))
            return patch
        #pad_x, pad_y = min(5,(bbox.x2-bbox.x1)//4), min(5,(bbox.y2-bbox.y1)//4)
        #x1, x2, y1, y2 = int(bbox.x1+pad_x), int(bbox.x2-pad_x), int(bbox.y1+pad_y), int(bbox.y2-pad_y)
        x1, x2, y1, y2 = int(bbox.x1), int(bbox.x2), int(bbox.y1), int(bbox.y2)
        target_patch = image[y1:y2, x1:x2, :]

        h, w = target_patch.shape[:2]
        if w<8 or h<8:
            return image
        case_p=np.random.random()
        #print(case_p)
        steps = 4
        if case_p>0.75: #horizontal shift
            step_h = h//steps
            step_shift = min(5,max(random.choice(np.arange(1, w // 2))//(steps-1),1))
            #print('step_shift '+ str(step_shift))
            target_patch[step_h:step_h * 2, :, :] = circshift(target_patch[step_h:step_h * 2, :, :], 0, step_shift)
            target_patch[step_h * 2:step_h * 3, :, :] = circshift(target_patch[step_h * 2:step_h * 3, :, :], 0, step_shift*2)
            target_patch[step_h * 3:step_h * 4, :, :] = circshift(target_patch[step_h * 3:step_h * 4, :, :], 0, step_shift*3)
            target_patch[step_h * 4:, :, :] = circshift(target_patch[step_h * 4:, :, :], 0, step_shift*4)
        elif case_p>0.5: #vertical shift
            step_h = w//steps
            step_shift = min(5,max(random.choice(np.arange(1, h // 2))//(steps-1),1))
            #print('step_shift ' + str(step_shift))
            target_patch[:,step_h:step_h * 2, :] = circshift(target_patch[:,step_h:step_h * 2, :], step_shift, 0)
            target_patch[:,step_h * 2:step_h * 3,  :] = circshift(target_patch[:,step_h * 2:step_h * 3,  :], step_shift*2, 0)
            target_patch[:,step_h * 3:step_h * 4,  :] = circshift(target_patch[:,step_h * 3:step_h * 4,  :], step_shift*3, 0)
            target_patch[:,step_h * 4:,  :] = circshift(target_patch[:,step_h * 4:,  :], step_shift*4, 0)
        else:
            s_down = random.choice(np.hstack((np.arange(1, h // 4), np.arange(h // 4 * 3, h - 1))))
            s_left = random.choice(np.hstack((np.arange(1, w // 4), np.arange(w // 4 * 3, w - 1))))
            # s_down = random.choice(np.hstack((np.arange( h // 4,h // 4 * 3))))
            # s_left = random.choice(np.hstack((np.arange(w // 4,w // 4 * 3))))
            #print('shift down, left: ' + str(s_down) + ' ' + str(s_left))
            target_patch = circshift(target_patch, s_down, s_left)

        image[y1:y2, x1:x2, :] = target_patch
        return image

    def intra_shift_intra(self, image, bbox):
        # perform intra-shift of the target
        # case1 only move part of the target; case2
        def circshift(patch, shiftnum_down, shiftnum_left):
            h, w = patch.shape[:2]
            patch = np.vstack((patch[(h - shiftnum_down):, :, :], patch[:(h - shiftnum_down), :, :]))
            patch = np.hstack((patch[:, (w - shiftnum_left):, :], patch[:, :(w - shiftnum_left), :]))
            return patch

        # pad_x, pad_y = min(5,(bbox.x2-bbox.x1)//4), min(5,(bbox.y2-bbox.y1)//4)
        # x1, x2, y1, y2 = int(bbox.x1+pad_x), int(bbox.x2-pad_x), int(bbox.y1+pad_y), int(bbox.y2-pad_y)
        h_pad, w_pad = max(5,int((bbox.y2-bbox.y1)/20)), max(5,int((bbox.x2-bbox.x1)/20))
        x1, x2, y1, y2 = int(bbox.x1+w_pad), int(bbox.x2-w_pad),int(bbox.y1+h_pad), int(bbox.y2-h_pad)
        w, h = x2-x1, y2-y1
        if w < 4 or h < 4:
            return image
        #case_p = np.random.random()
        #print('w h: '+str(w)+' ' +str(h))
        target_patch = image[y1:y2, x1:x2, :]
        # print(case_p)
        steps = 4
        # if case_p > 0.75:  # horizontal shift
        #     step_h = h // steps
        #     step_shift = min(5, max(random.choice(np.arange(1, w // 2)) // (steps - 1), 1))
        #     # print('step_shift '+ str(step_shift))
        #     target_patch[step_h:step_h * 2, :, :] = circshift(target_patch[step_h:step_h * 2, :, :], 0, step_shift)
        #     target_patch[step_h * 2:step_h * 3, :, :] = circshift(target_patch[step_h * 2:step_h * 3, :, :], 0,
        #                                                           step_shift * 2)
        #     target_patch[step_h * 3:step_h * 4, :, :] = circshift(target_patch[step_h * 3:step_h * 4, :, :], 0,
        #                                                           step_shift * 3)
        #     target_patch[step_h * 4:, :, :] = circshift(target_patch[step_h * 4:, :, :], 0, step_shift * 4)
        # elif case_p > 0.5:  # vertical shift
        #     step_h = w // steps
        #     step_shift = min(5, max(random.choice(np.arange(1, h // 2)) // (steps - 1), 1))
        #     # print('step_shift ' + str(step_shift))
        #     target_patch[:, step_h:step_h * 2, :] = circshift(target_patch[:, step_h:step_h * 2, :], step_shift, 0)
        #     target_patch[:, step_h * 2:step_h * 3, :] = circshift(target_patch[:, step_h * 2:step_h * 3, :],
        #                                                           step_shift * 2, 0)
        #     target_patch[:, step_h * 3:step_h * 4, :] = circshift(target_patch[:, step_h * 3:step_h * 4, :],
        #                                                           step_shift * 3, 0)
        #     target_patch[:, step_h * 4:, :] = circshift(target_patch[:, step_h * 4:, :], step_shift * 4, 0)
        # else:
        s_down = random.choice(np.hstack((np.arange(1, h // 4), np.arange(h // 4 * 3, h - 1))))
        s_left = random.choice(np.hstack((np.arange(1, w // 4), np.arange(w // 4 * 3, w - 1))))
        case_p = np.random.random()
        if case_p>0.75:
            s_left = 0
        elif case_p<0.5:
            s_down = 0
            # s_down = random.choice(np.hstack((np.arange( h // 4,h // 4 * 3))))
            # s_left = random.choice(np.hstack((np.arange(w // 4,w // 4 * 3))))
            # print('shift down, left: ' + str(s_down) + ' ' + str(s_left))
        target_patch = circshift(target_patch, s_down, s_left)

        image[y1:y2, x1:x2, :] = target_patch
        return image

    def crop_paste_object_template(self, image, target_img, target_box, pad_sz=[10,10]): # pad_sz [w h]
        # crop the target patch and paste it onto the base image at a random location
        H, W, _ = image.shape
        paste_pad_x, paste_pad_y = 10, 10
        w_in,h_in = target_box.x2 - target_box.x1+1, target_box.y2 - target_box.y1+1

        pad_sz[1], pad_sz[0] = min(10,max(1,round(h_in/20))), min(10,max(1,round(w_in/20)))
        h, w = h_in + 2 * pad_sz[1], w_in + 2 * pad_sz[1]
        x_pad_ratio,y_pad_ratio = pad_sz[0]/w,pad_sz[1]/h
        object_h, object_w = h, w
        re_scale = max(3 * h / H, 3 * w / W)
        if re_scale>1:
            object_h, object_w = round(h/re_scale), round(w/re_scale)

        object_w = max(2 * pad_sz[0]+20, object_w)
        object_h = max(2 * pad_sz[1]+20, object_h)

        # First, get the random locations to paste at
        xs = random.choice(range(1, W-object_w-paste_pad_x))
        ys = random.choice(range(1, H-object_h-paste_pad_y))
        # Second, perform the pasting
        def gen_mask(p_sz, pad_sz=[5,5]):  # p_sz: size of the patch

            pw, ph = p_sz
            pad_sz[0], pad_sz[1] = min(pad_sz[0],pw//2), min(pad_sz[1],ph//2)
            if pw<2 or ph<2:
                print(pw)
            h_inds = np.zeros(ph, dtype=np.float32)
            w_inds = np.zeros(pw, dtype=np.float32)
            w_inds[:pad_sz[0]] = np.arange(pad_sz[0] , 0, -1)
            w_inds[-pad_sz[0]:] = np.arange(1, pad_sz[0]+1)
            h_inds[:pad_sz[1]] = np.arange(pad_sz[1] , 0, -1)
            h_inds[-pad_sz[1]:] = np.arange(1, pad_sz[1]+1)

            h_inds = np.tile(h_inds.reshape(-1, 1) * 1.0 / pad_sz[1], (1, pw))
            w_inds = np.tile((w_inds * 1.0 / pad_sz[0]), (ph, 1))
            # inds,_ = torch.max(torch.cat((h_inds.reshape(1,-1),w_inds.reshape(1,-1)),0),0)
            inds = (h_inds + w_inds)
            inds = np.where(inds > 1, np.ones_like(inds), inds)

            ind_h,ind_w = inds.shape
            return inds

        target_patch = self._crop_roi_new(target_img, (target_box.x1-pad_sz[0], target_box.y1-pad_sz[1], \
                                                 target_box.x2+pad_sz[0], target_box.y2+pad_sz[1]), (object_w, object_h), padding=(0, 0, 0))
        new_pad_x, new_pad_y = max(1,round(object_w*x_pad_ratio)), max(1,round(object_h*y_pad_ratio))
        if new_pad_x<1 or new_pad_y<1:
            print(new_pad_x)
        mask = np.expand_dims(gen_mask((object_w, object_h),[new_pad_x, new_pad_y]), 2)

        image[ys:ys+object_h, xs:xs+object_w,:] = mask * image[ys:ys+object_h, xs:xs+object_w,:] +(1-mask) * target_patch
        target_box = [xs+new_pad_x, ys+new_pad_y, object_w-2*new_pad_x, object_h-2*new_pad_y]
        return image, torch.Tensor(target_box), target_patch.copy()  # [x,y,w,h]

    def crop_paste_object_template_nopad(self, image, target_img, target_box, pad_sz=[10,10]): # pad_sz [w h]
        # crop the target patch and paste it onto the base image at a random location
        H, W, _ = image.shape
        paste_pad_x, paste_pad_y = 10, 10
        object_w,object_h = target_box.x2 - target_box.x1+1, target_box.y2 - target_box.y1+1

        re_scale = max(3 * object_h / H, 3 * object_w / W)
        if re_scale>1:
            object_h, object_w = round(object_h/re_scale), round(object_w/re_scale)

        object_w = max(20, object_w)
        object_h = max(20, object_h)

        # First, get the random locations to paste at
        xs = random.choice(range(1, W-object_w-paste_pad_x))
        ys = random.choice(range(1, H-object_h-paste_pad_y))
        # Second, perform the pasting
        target_patch = self._crop_roi_new(target_img, (target_box.x1, target_box.y1, \
                                                 target_box.x2, target_box.y2), (object_w, object_h), padding=(0, 0, 0))

        image[ys:ys+object_h, xs:xs+object_w,:] = target_patch
        target_box = [xs, ys, object_w, object_h]
        return image, torch.Tensor(target_box), target_patch.copy()  # [x,y,w,h]

    def crop_paste_object_template_nopad_shear(self, image, target_img, target_box, pad_sz=[10,10]): # pad_sz [w h]
        # crop the target patch and paste it onto the base image at a random location
        H, W, _ = image.shape
        paste_pad_x, paste_pad_y = 10, 10
        object_w,object_h = target_box.x2 - target_box.x1+1, target_box.y2 - target_box.y1+1

        re_scale = max(3 * object_h / H, 3 * object_w / W)
        if re_scale>1:
            object_h, object_w = round(object_h/re_scale), round(object_w/re_scale)

        object_w = max(20, object_w)
        object_h = max(20, object_h)

        # First, get the random locations to paste at
        xs = random.choice(range(1, W-object_w-paste_pad_x))
        ys = random.choice(range(1, H-object_h-paste_pad_y))
        # Second, perform the pasting
        target_patch = self._crop_roi_new(target_img, (target_box.x1, target_box.y1, \
                                                 target_box.x2, target_box.y2), (object_w, object_h), padding=(0, 0, 0))

        image[ys:ys+object_h, xs:xs+object_w,:] = target_patch.copy()
        target_box = [xs, ys, object_w, object_h]

        image, target_box = self.affine_aug(image, Corner(xs, ys, xs+object_w-1, ys+object_h-1))

        return image, torch.Tensor([target_box.x1, target_box.y1, target_box.x2-target_box.x1+1, target_box.y2-target_box.y1+1]), target_patch  # [x,y,w,h]

    def crop_rescale_paste(self, image, target_img, target_box, scale, pad_sz=[10,10]): # pad_sz [w h]
        # crop the target patch and paste it onto the base image at a random location
        H, W, _ = image.shape
        paste_pad_x, paste_pad_y = 10, 10
        w_in, h_in = target_box.x2 - target_box.x1 + 1, target_box.y2 - target_box.y1 + 1

        pad_sz[1], pad_sz[0] = min(10, max(1, round(h_in / 20))), min(10, max(1, round(w_in / 20)))
        h, w = h_in + 2 * pad_sz[1], w_in + 2 * pad_sz[1]
        #x_pad_ratio, y_pad_ratio = pad_sz[0] / w, pad_sz[1] / h
        object_h, object_w = h, w
        re_scale = max(3 * h / H, 3 * w / W)

        if re_scale>1:
            object_h, object_w = round(h/re_scale), round(w/re_scale)

        if scale:
            scale_x = (1.0 + Augmentation_CTP.random() * self.scale)
            scale_y = (1.0 + Augmentation_CTP.random() * self.scale)
            object_w, object_h = round(scale_x*object_w), round(scale_y*object_h)

        object_w = max(2 * pad_sz[0]+20, int(object_w))
        object_h = max(2 * pad_sz[1]+20, int(object_h))

        x_pad_ratio = pad_sz[0]/(target_box.x2-target_box.x1+1+2*pad_sz[0])
        y_pad_ratio = pad_sz[1]/(target_box.y2-target_box.y1+1+2*pad_sz[1])

        # First, get the random locations to paste at
        xs = random.choice(range(1, int(W-object_w-paste_pad_x)))
        ys = random.choice(range(1, int(H-object_h-paste_pad_y)))
        # Second, perform the pasting
        def gen_mask(p_sz, pad_sz=[5,5]):  # p_sz: size of the patch

            pw, ph = p_sz
            pad_sz[0], pad_sz[1] = min(pad_sz[0],pw//2), min(pad_sz[1],ph//2)
            if pw<2 or ph<2:
                print(pw)
            h_inds = np.zeros(ph, dtype=np.float32)
            w_inds = np.zeros(pw, dtype=np.float32)
            w_inds[:pad_sz[0]] = np.arange(pad_sz[0] , 0, -1)
            w_inds[-pad_sz[0]:] = np.arange(1, pad_sz[0]+1)
            h_inds[:pad_sz[1]] = np.arange(pad_sz[1] , 0, -1)
            h_inds[-pad_sz[1]:] = np.arange(1, pad_sz[1]+1)

            h_inds = np.tile(h_inds.reshape(-1, 1) * 1.0 / pad_sz[1], (1, pw))
            w_inds = np.tile((w_inds * 1.0 / pad_sz[0]), (ph, 1))
            # inds,_ = torch.max(torch.cat((h_inds.reshape(1,-1),w_inds.reshape(1,-1)),0),0)
            inds = (h_inds + w_inds)
            inds = np.where(inds > 1, np.ones_like(inds), inds)

            ind_h,ind_w = inds.shape
            return inds

        target_patch = self._crop_roi_new(target_img, (target_box.x1-pad_sz[0], target_box.y1-pad_sz[1], \
                                                 target_box.x2+pad_sz[0], target_box.y2+pad_sz[1]), (object_w, object_h), padding=(0, 0, 0))
        new_pad_x, new_pad_y = max(1,round(object_w*x_pad_ratio)), max(1,round(object_h*y_pad_ratio))
        if new_pad_x<1 or new_pad_y<1:
            print(new_pad_x)
        mask = np.expand_dims(gen_mask((object_w, object_h),[new_pad_x, new_pad_y]), 2)

        image[ys:ys+object_h, xs:xs+object_w,:] = mask * image[ys:ys+object_h, xs:xs+object_w,:] +(1-mask) * target_patch

        return image, Corner(xs+new_pad_x, ys+new_pad_y, xs+object_w-new_pad_x, ys+object_h-new_pad_y)  # [x,y,w,h]

    def crop_rescale_paste_nopad(self, image, target_patch, scale, pad_sz=[10,10]): # pad_sz [w h]
        # crop the target patch and paste it onto the base image at a random location
        H, W, _ = image.shape
        paste_pad_x, paste_pad_y = 10, 10
        object_h, object_w = target_patch.shape[:2]
        #x_pad_ratio, y_pad_ratio = pad_sz[0] / w, pad_sz[1] / h

        # re_scale = max(3 * object_h / H, 3 * object_w / W)
        # if re_scale>1:
        #     object_h, object_w = round(object_h/re_scale), round(object_w/re_scale)

        if scale:
            scale_x = (1.0 + Augmentation_CTP.random() * self.scale)
            scale_y = (1.0 + Augmentation_CTP.random() * self.scale)
            object_w_new, object_h_new = round(scale_x*object_w), round(scale_y*object_h)

        # object_w = max(20, int(object_w))
        # object_h = max(20, int(object_h))

        # First, get the random locations to paste at
        xs = random.choice(range(1, int(W-object_w_new-paste_pad_x)))
        ys = random.choice(range(1, int(H-object_h_new-paste_pad_y)))
        # Second, perform the pasting

        # target_patch = self._crop_roi_new(target_patch, (0, 0, \
        #                                           object_w, object_h), (object_w_new, object_h_new), padding=(0, 0, 0))


        image[ys:ys+object_h_new, xs:xs+object_w_new,:] = cv2.resize(target_patch,(object_w_new,object_h_new))

        return image, Corner(xs, ys, xs+object_w_new, ys+object_h_new)  # [x,y,w,h]

    def __call__(self, test_image, target_patch):
        # crop and paste with scale transformation
        img_syn, target_box = self.crop_rescale_paste_nopad(test_image, target_patch, self.scale)
       # img_syn, target_box = self.crop_rescale_paste(test_image, target_patch, target_box,self.scale)
        if self.shear>np.random.random():
            img_syn, target_box = self.affine_aug(img_syn, target_box)
        # use blur, occ and IS separately
        if self.blur_occ_IS > np.random.random():
            case_p = np.random.random()
            if self.blur > case_p:
                img_syn, blur_pa = self._blur_aug(img_syn)
                #print('blur '+str(case_p))
            elif self.occ > case_p:
                img_syn = self.occ_mask(img_syn, target_box)
                #print('occ ' + str(case_p))
            else:
                img_syn = self.intra_shift(img_syn, target_box)
                # cfg.DATASET.SEARCH.ISI
                #template_image = self.template_aug.intra_shift_intra(template_image, template_box)
                #print('IS ' + str(case_p))
        # flip augmentation
        if self.flip and self.flip > np.random.random():
                img_syn, target_box = self._flip_aug(img_syn, target_box)

        return img_syn, torch.Tensor([target_box.x1, target_box.y1, target_box.x2-target_box.x1+1, target_box.y2-target_box.y1+1]) # [x_tl,y_tl,w,h]


if __name__ == '__main__':
    test_aug = Augmentation_CTP(
            0,  # cfg.DATASET.SEARCH.SHIFT,
            0.3,  # cfg.DATASET.SEARCH.SCALE,
            0.9,  # cfg.DATASET.SEARCH.BLUR,
            0.0,  # cfg.DATASET.SEARCH.FLIP,
            0.0,  # cfg.DATASET.SEARCH.COLOR,
            0.0,  # cfg.DATASET.SEARCH.OCC,
            0.0,  # cfg.DATASET.SEARCH.SHEAR
    )
    # #img63= cv2.imread('./0001.jpg')  #  134,55,60,88 #Center(164, 99, 60,88)
    # #box63 = center2corner(Center(164, 99, 60,88))
    #
    # #img1 = cv2.imread('/home/data/xin/projects/pysot/pysot/datasets/0001.jpg')
    # # box1 = Corner(124, 45, 203, 153)
    # # img1 = cv2.imread('/home/data/testing_dataset/OTB100/Dog/img/0022.jpg')
    # # box1 = Corner(60,83,54+60-1,79+88-1)
    # # img1 = cv2.imread('/home/data/testing_dataset/OTB100/Bird2/img/0001.jpg')
    # # box1 = Corner(82,218,82+69-1,218+73-1) # 82,218,69,73
    # img1 = cv2.imread('/home/data/testing_dataset/OTB100/Bird2/img/0022.jpg')
    # box1 = Corner(175, 229, 175+69-1, 234+73-1) #175,234,69,73
    # #img63 = cv2.imread('/home/data/xin/projects/pysot/pysot/datasets/0063.jpg')
    # img63 = cv2.imread('/home/data/testing_dataset/OTB100/Bird2/img/0050.jpg')
    # #img63 = cv2.imread('/home/data/testing_dataset/OTB100/Dog/img/0001.jpg')
    # #cv2.rectangle(img1, (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), (0, 255, 0))
    # #target, target_bbox = crop_with_pad(img1, box1, 5)
    #
    # #img3, bbox=test_aug(img63, img1[int(box1[1]):int(box1[3]),int(box1[0]):int(box1[2]),:])
    # #img3, bbox = test_aug(img63, img1,box1)
    # img3, bbox, target_patch = test_aug.crop_paste_object_template(img63, img1, box1, [10, 10])
    # # img3 = cv2.imread('./0044.jpg')#  61 173 46 25
    # # bbox = center2corner(Center(84,185,46,25))
    # # img3 = cv2.imread('./0208.jpg')  #      60      129     241     98
    # # bbox = center2corner(Center(180,178,241,98))
    # # cv2.rectangle(img3,(60,129),(301,227),(255,0,0))
    #
    # img3,bbox= test_aug.affine_aug(img3,target_patch)
    # # img3, bbox=test_aug(img3,bbox,255)
    # #img3=test_aug.intra_shift(img3,bbox)
    # #img3=test_aug.intra_shift_intra(img3,bbox)
    # # # affine on the template
    # # img3, bbox = test_aug.affine_aug(image=img3, bbox=box3) # bbox(tensor), bbs (tl,sz)
    # # # crop-paste
    # # img3, bbox = gen_ss_sample_new(img3, [int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)], img63)  # [195, 214, 228, 294]
    # # # box_tmp = center2corner(Center((bbox.x1+bbox.x2)/2,(bbox.y1+bbox.y2)/2,34,81))
    # # # cv2.rectangle(img3, (int(box_tmp[0]), int(box_tmp[1])), (int(box_tmp[2]), int(box_tmp[3])), (255, 0, 0))
    # #
    # # # shift and scale
    # # img3, bbox = test_aug._shift_scale_aug(img3, bbox, center2corner(Center(212, 254, 254, 254)), 255)
    # # # crop
    # # img3 = test_aug._crop_aug(img3, bbox)
    # # # blur
    # # img3 = test_aug._blur_moco_aug(img3)
    # # #flip
    # # img3, bbox = test_aug._flip_aug(img3, bbox)
    # # # color jitter
    # # img3= test_aug._color_jitter_aug(img3)
    # cv2.rectangle(img3, (int(bbox[0]), int(bbox[1])), (int(bbox[2])+int(bbox[0]), int(bbox[3])+int(bbox[1])), (0, 0, 255))
    # cv2.imshow('2', img3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




    # img3 = test_aug.toTensor(img3)
    # img3 = test_aug.normalize(img3)


# #-affine
#     affine = tfm.Transform( tfm.RandomAffine(max_rotation=10,max_shear=0.15,
#                                                 max_scale=0.3,max_ar_factor=0),)
#     img3, bbs = affine(image=img63, bbox=bbox)#coords=coord) # bbox(tensor), bbs (tl,sz)
# #-crop-paste (cross-video)
#     img3, bbox = gen_ss_sample_new(img3, [int(bbs[0]), int(bbs[1]), int(bbs[0] + bbs[2] - 1), int(bbs[1] + bbs[3] - 1)],
#                                img13)  # [195, 214, 228, 294]
# #-random crop
#     img3= test_aug._crop_aug(img3, bbox,  thres=0.7)
# #-blur
#     img3 = test_aug._blur_moco_aug(img3) # set to 0.1 is enough
# #-flip
#     img3, bbox = test_aug._flip_aug(img3, bbox)
#     crop_bbox = center2corner(Center(212, 254,  254, 254))
# #-shift-scale
#     img3, bbox = test_aug._shift_scale_aug(img3, bbox, crop_bbox, 255)
#     # the following should be consistent with the online tracking part
# #-colorjitter
#     img3 = test_aug.to_pil(cv2.cvtColor(img3,cv2.COLOR_RGB2BGR))
#     img3 = test_aug.color_jitter(img3)
#     img3 = cv2.cvtColor(img3,cv2.COLOR_RGB2BGR)
#
# #-normalization
#     img3 = test_aug.toTensor(img3)
#     img3 = test_aug.normalize(img3)
