import os

import numpy as np

from PIL import Image
import tensorflow as tf
import matplotlib.image as mpimg


# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()


# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    print(str(max_labels) + ' ' + str(max_predictions))


# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx][0] > 0.5:  # bgrd
                l = 0
            else:
                l = 1
            array_labels[j:j + w, i:i + h] = l
            idx = idx + 1
    return array_labels


def img_float_to_uint8(img, pixel_depth):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * pixel_depth).round().astype(np.uint8)
    return rimg


# Extract patches from a given image
def img_crop(im, w, h, padding):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_gt = len(im.shape) < 3
    for i in range(padding, imgheight + padding, h):
        for j in range(padding, imgwidth + padding, w):
            if is_gt:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j - padding:j + w + padding, i - padding:i + h + padding, :]
            list_patches.append(im_patch)
    return list_patches


def pad_image(img, padding_size):
    """
    Extend the canvas of an image. Mirror boundary conditions are applied.
    """
    if len(img.shape) < 3:
        # Greyscale image (ground truth)
        data = np.pad(img, ((padding_size, padding_size), (padding_size, padding_size)), 'reflect')
    else:
        # RGB image
        data = np.pad(img, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), 'reflect')
    return data


def load_image(filename):
    if os.path.isfile(filename):
        img = mpimg.imread(filename)
        return img
    else:
        raise Exception('File ' + filename + ' does not exist')


def load_images(train_dir, gt_dir, num_images):
    print("loading images from disk .. ")
    imgs = np.asarray([load_image(train_dir + "satImage_%.4d" % i + ".png") for i in range(1, num_images + 1)])
    gt_imgs = np.asarray([load_image(gt_dir + "satImage_%.4d" % i + ".png") for i in range(1, num_images + 1)])
    print("finished loading images from disk ..")
    print("imgs shape : ", end=' ')
    print(imgs.shape)
    return imgs, gt_imgs


def gen_patches(imgs, window_size, patch_size):
    padding_size = int((window_size - patch_size) / 2)

    img_patches = np.asarray(
        [img_crop(imgs[i], patch_size, patch_size, padding_size) for i in range(imgs.shape[0])])

    if len(img_patches.shape) > 3:
        img_patches = img_patches.reshape((-1, img_patches.shape[2], img_patches.shape[3], img_patches.shape[4]))
    else:
        img_patches = img_patches.reshape((-1, img_patches.shape[2], img_patches.shape[3]))

    return img_patches


def preprocess_imgs(imgs, window_size, patch_size):
    padding_size = int((window_size - patch_size) / 2)
    padded_imgs = np.asarray([pad_image(imgs[i], padding_size) for i in range(imgs.shape[0])])

    return padded_imgs.astype('float32')/255
