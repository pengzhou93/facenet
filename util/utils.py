import scipy.misc
import numpy as np
import os
import scipy.misc
import tensorflow as tf

    
class ScopeData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)

def transform(image, npx=128, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def doresize(x, rows, cols):
    y = scipy.misc.imresize(x, [rows, cols])
    return y

def center_crop(x, crop_h, crop_w=None, resize_w=128):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def test_batch_seq_images(images):
    batch_size, seq_length, _, _, _ = images.shape
    images = images.reshape([-1] + list(images.shape[2:]))
    sum_img = merge(images, (batch_size, seq_length))
    # scipy.misc.imshow(sum_img)
    # assert 0, "test triplet batch images"
    return sum_img


def test_batch_images(images, num_images = 30, cols = 6):
    # batch_size, _, _, _ = images.shape
    # if batch_size > num_images:
    #     batch_size = num_images
    #     images = images[0:num_images,]
    size = (num_images // cols, cols)
    sum_img = merge(images, size)
    return sum_img

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img
