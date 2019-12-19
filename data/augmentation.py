# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image

# Helper functions

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
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


def add_noise(ratio, img):
    #INPUT: ratio of noise vs. clean pixels and the image
    
    #make a copy not to change the original and get the size
    img_copy = img.copy()
    size = img.shape
    #salt-and-paper noise amount
    amount = int(size[0]*size[1]*ratio)
    #loop through the randomly change a pixel to either black or white
    for i in range(amount):
        point = [np.random.randint(0, size[0], amount) for i in size[:2]]
        img_copy[point[0], point[1], :] = np.random.choice(2)

    return img_copy

def lighting(img):
    #INPUT: image with 3 channels
    #Returns same image with Gaussian shading
    
    #get the size of the image
    size = img.shape
    
    #generate a matrix of the same size with random numbers in [0-1) and round
    rand = np.round(np.random.random((size[0], size[1], 1)),7)
    #make it have the same dimension (number of channels) with the image
    gaussian = np.concatenate((rand, rand, rand), axis = 2)
    #add the gaussian image to the normal image with weight 0.75 and 0.25 respectively
    new_img = img*0.75 + gaussian*0.25*0.25

    
    return new_img

# Loaded a set of images
root_dir = "../data/training/"
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = 10 # Load maximum 20 images
print("Loading " + str(n) + " images")
imgs = [load_image(image_dir + files[i]) for i in range(n)]
gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " images")
gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
j=n+1
for i in range(n):
    #load images
    img = imgs[i]
    gt_img = gt_imgs[i]
    #set 90degree rotation directories for images and groundtruths
    rot90_dir = root_dir + "images/"
    rot90_dir_gt = root_dir + "groundtruth/"
    #rotate images
    img = np.rot90(img)
    gt_img = np.rot90(gt_img)
    #save images
    plt.imsave(rot90_dir +'satImage_'+str(j).zfill(4)+'.png', img)
    plt.imsave(rot90_dir_gt +'satImage_'+str(j).zfill(4)+'.png', gt_img, cmap='Greys_r')
    j+=1
    
    #set 180degree rotation directories for images and groundtruths
    rot180_dir = root_dir + "images/"
    rot180_dir_gt = root_dir + "groundtruth/"
    #rotate images
    img = np.rot90(img)
    gt_img = np.rot90(gt_img)
    #save images
    plt.imsave(rot180_dir +'satImage_'+str(j).zfill(4)+'.png', img)
    plt.imsave(rot180_dir_gt +'satImage_'+str(j).zfill(4)+'.png', gt_img, cmap='Greys_r')
    j+=1
    
    #set 270degree rotation directories for images and groundtruths
    rot270_dir = root_dir + "images/"
    rot270_dir_gt = root_dir + "groundtruth/"
    #rotate images
    img = np.rot90(img)
    gt_img = np.rot90(gt_img)
    #save images
    plt.imsave(rot270_dir +'satImage_'+str(j).zfill(4)+'.png', img)
    plt.imsave(rot270_dir_gt +'satImage_'+str(j).zfill(4)+'.png', gt_img, cmap='Greys_r')
    j+=1
    
    #set 90 degree mirror directories for images and groundtruths
    mir90_dir = root_dir + "images/"
    mir90_dir_gt = root_dir + "groundtruth/"
    #rotate images to get raw images
    img = np.rot90(img)
    gt_img = np.rot90(gt_img)
    #save images with fliping wrt to vertival axis
    plt.imsave(mir90_dir +'satImage_'+str(j).zfill(4)+'.png', img)
    plt.imsave(mir90_dir_gt +'satImage_'+str(j).zfill(4)+'.png', gt_img, cmap='Greys_r')
    j+=1
    
    #set 180 degree mirror directories for images and groundtruths
    mir180_dir = root_dir + "images/"
    mir180_dir_gt = root_dir + "groundtruth/"
    #save images with fliping wrt to vertival axis
    plt.imsave(mir180_dir +'satImage_'+str(j).zfill(4)+'.png', img)
    plt.imsave(mir180_dir_gt +'satImage_'+str(j).zfill(4)+'.png', gt_img, cmap='Greys_r')
    j+=1
    
    #set 45 degree mirror directories for images and groundtruths
    mir45_dir = root_dir + "images/"
    mir45_dir_gt = root_dir + "groundtruth/"
    #save images with fliping wrt to vertival axis
    plt.imsave(mir45_dir +'satImage_'+str(j).zfill(4)+'.png', img)
    plt.imsave(mir45_dir_gt +'satImage_'+str(j).zfill(4)+'.png', gt_img, cmap='Greys_r')
    j+=1
    
    #set 135 degree mirror directories for images and groundtruths
    mir135_dir = root_dir + "images/"
    mir135_dir_gt = root_dir + "groundtruth/"
    #save images with fliping wrt to vertival axis
    plt.imsave(mir135_dir +'satImage_'+str(j).zfill(4)+'.png', img)
    plt.imsave(mir135_dir_gt +'satImage_'+str(j).zfill(4)+'.png', gt_img, cmap='Greys_r')
    j+=1
    
    #set salt-peper noise directories for images and groundtruths
    noise_dir = root_dir + "images/"
    noise_dir_gt = root_dir + "groundtruth/"
    #save images
    plt.imsave(noise_dir +'satImage_'+str(j).zfill(4)+'.png', add_noise(0.0003, img))
    plt.imsave(noise_dir_gt+'satImage_'+str(j).zfill(4)+'.png', gt_img, cmap='Greys_r')
    j+=1

    #set lighting noise directories for images and groundtruths\n",
    light_dir = root_dir + "images/"
    light_dir_gt = root_dir + "groundtruth/"
    #save images\n",
    plt.imsave(light_dir +'satImage_'+str(j).zfill(4)+'.png', lighting(img))
    plt.imsave(light_dir_gt +'satImage_'+str(j).zfill(4)+'.png', gt_img, cmap='Greys_r')
    j+=1
    