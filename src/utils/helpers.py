import os

import numpy as np

from PIL import Image
import tensorflow as tf
import matplotlib.image as mpimg
import re


def img_crop(im, w, h, stride, padding):
    """ Crop an image into patches, taking into account mirror boundary conditions. """
    assert len(im.shape) == 3, 'Expected RGB image.'
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    im = np.lib.pad(im, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
    for i in range(padding, imgheight + padding, stride):
        for j in range(padding, imgwidth + padding, stride):
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


def group_patches(patches, num_images):
    return patches.reshape(num_images, -1)


def gen_patches(imgs, window_size, patch_size):
    padding_size = int((window_size - patch_size) / 2)

    patches = np.asarray([img_crop(imgs[i], patch_size, patch_size,patch_size ,padding_size) for i in range(imgs.shape[0])])
    print(patches.shape)

    return patches.reshape(-1, patches.shape[2], patches.shape[3], patches.shape[4])


def preprocess_imgs(imgs):
    
    return imgs.astype(float)/255.0


def generate_submission(model, submission_filename, *image_filenames):
    """ Generate a .csv containing the classification of the test set. """
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(model, fn))


# Reads an image and  outputs the label that should go into the submission file
def mask_to_submission_strings(model, filename):
    img_number = int(re.search(r"\d+", filename).group(0))
    image = load_image(filename)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    labels = model.predict(image)
    labels = labels.reshape(-1)
    patch_size = model.patch_size
    count = 0
    print("Processing image => " + filename)
    for j in range(0, image.shape[2], patch_size):
        for i in range(0, image.shape[1], patch_size):
            label = int(labels[count])
            count += 1
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))
