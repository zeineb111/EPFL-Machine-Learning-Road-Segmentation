# Helper functions
import os

import matplotlib.image as mpimg
import numpy as np

PIXEL_DEPTH = 255


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)
    return list_patches


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg


def load_data(data_path, test):
    files = os.listdir(data_path)
    n = len(files)

    if test:
        imgs = [load_image(data_path + '/' + files[i]) for i in range(n)]
    else:
        imgs = [load_image(data_path + files[i]) for i in range(n)]

    return np.asarray(imgs)


def img_predict(img, model):
    width = img.shape[0]
    height = img.shape[1]
    img1 = img[:400, :400]
    img2 = img[:400, -400:]
    img3 = img[-400:, :400]
    img4 = img[-400:, -400:]

    imgs = np.array([img1, img2, img3, img4])
    predictions = model.predict(imgs)

    prediction = np.zeros((width, height, 1))

    prediction[:400, :400] = predictions[0]
    prediction[:400, -400:] = predictions[1]
    prediction[-400:, :400] = predictions[2]
    prediction[-400:, -400:] = predictions[3]

    return prediction
