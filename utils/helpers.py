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


def concatenate_images(img, gt_img, pixel_depth):
    n_channels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if n_channels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img, pixel_depth)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img, pixel_depth)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def make_img_overlay(img, predicted_img, pixel_depth):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * pixel_depth

    img8 = img_float_to_uint8(img, pixel_depth)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


# Make an image summary for 4d tensor image with index idx
def get_image_summary(img, pixel_depth, idx=0):
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    img_w = img.get_shape().as_list()[1]
    img_h = img.get_shape().as_list()[2]
    min_value = tf.reduce_min(V)
    V = V - min_value
    max_value = tf.reduce_max(V)
    V = V / (max_value * pixel_depth)
    V = tf.reshape(V, (img_w, img_h, 1))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, (-1, img_w, img_h, 1))
    return V


# Make an image summary for 3d tensor image with index idx
def get_image_summary_3d(img):
    V = tf.slice(img, (0, 0, 0), (1, -1, -1))
    img_w = img.get_shape().as_list()[1]
    img_h = img.get_shape().as_list()[2]
    V = tf.reshape(V, (img_w, img_h, 1))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, (-1, img_w, img_h, 1))
    return V


# Extract patches from a given image
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


def img_crop_windows(im, window_size, patch_size):
    # padding
    padding_size = int((window_size - patch_size) / 2)
    padded_im = np.pad(im, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)))  # pad with 0s

    width = im.shape[0]
    height = im.shape[1]

    list_windows = []
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            im_window = padded_im[j:j + window_size, i:i + window_size, :]
            list_windows.append(im_window)
    return list_windows


def extract_training_data(train_dir, gt_dir, num_images, img_patch_size, img_window_size):
    imgs = []
    imgs_gt = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.4d" % i
        tr_image_filename = train_dir + imageid + ".png"
        gt_image_filename = gt_dir + imageid + ".png"
        if os.path.isfile(tr_image_filename):
            print('Loading ' + tr_image_filename)
            img = mpimg.imread(tr_image_filename)
            imgs.append(img)
        else:
            print('File ' + tr_image_filename + ' does not exist')
        if os.path.isfile(gt_image_filename):
            print('Loading ' + gt_image_filename)
            img_gt = mpimg.imread(gt_image_filename)
            imgs_gt.append(img_gt)
        else:
            print('File ' + gt_image_filename + ' does not exist')

    num_images = len(imgs)

    windows = [img_crop_windows(imgs[i], img_window_size, img_patch_size) for i in range(num_images)]
    windows_data = [windows[i][j] for i in range(len(windows)) for j in range(len(windows[i]))]
    patches_gt = [img_crop(imgs[i], img_patch_size, img_patch_size) for i in range(num_images)]
    patches_gt_data = [patches_gt[i][j] for i in range(len(patches_gt)) for j in range(len(patches_gt[i]))]
    labels = np.asarray([value_to_class(np.mean(patches_gt_data[i])) for i in range(len(patches_gt_data))])
    return np.asarray(windows_data), labels.astype(np.float32)


def extract_data(filename, num_images, img_patch_size):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.4d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / img_patch_size) * (IMG_HEIGHT / img_patch_size)

    img_patches = [img_crop(imgs[i], img_patch_size, img_patch_size) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return np.asarray(data)


# Extract label images
def extract_labels(filename, num_images, img_patch_size):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.4d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], img_patch_size, img_patch_size) for i in range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)


def balance_data(train_data, train_labels):
    c0 = 0  # bgrd
    c1 = 0  # road
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print(len(new_indices))
    print(train_data.shape)
    train_data = train_data[new_indices, :, :, :]
    train_labels = train_labels[new_indices]

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))
