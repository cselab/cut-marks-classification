#!/usr/bin/env python

# author: Wonmin Byeon (wonmin.byeon@gmail.com)
# Some functions are adapted by: https://github.com/cydonia999/Grad-CAM-in-TensorFlow/blob/master/grad-cam-tf.py

from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np
import sys
import os
import cv2
import argparse

import matplotlib.pyplot as plt
import math

import scipy.misc

__author__ = "Wonmin Byeon"

__maintainer__ = "Wonmin Byeon"
__email__ = "wonmin.byeon@gmail.com"

def grad_cam(reduced_loss, layer_name, sess, feed_dict):
    """
    calculate Grad-CAM
    """
    conv_output = sess.graph.get_tensor_by_name(layer_name + ':0')
    grads = tf.gradients(reduced_loss, conv_output)[0] # d loss / d conv
    output, grads_val = sess.run([conv_output, grads], feed_dict=feed_dict)
    weights = np.mean(grads_val, axis=(1, 2)) # average pooling
    cams = np.sum(weights * output, axis=3)
    return cams


def save_cam(cams, class_id, pred_class, gt_class, prob, image, image_path):
    """
    save Grad-CAM images
    """
    cam = cams[0] # the first GRAD-CAM for the first image in  batch
    image = np.uint8(image[0, :, :, 0] * 255.0) # RGB -> BGR
    h, w = image.shape
    cam = cv2.resize(cam, (w, h)) # enlarge heatmap
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam) # normalize
    image_ch3 = np.stack((image,)*3, axis=-1)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET) # balck-and-white to color
    cam = np.float32(cam) + np.float32(image_ch3) # everlay heatmap onto the image
    cam = 255 * cam / np.max(cam)
    cam = np.uint8(cam)

    # create image file names
    base_path, ext = os.path.splitext(image_path)
    base_path_class = "{}_{}_pred{}_gt{}_prob{:.3f}".format(base_path, str(class_id), str(pred_class), str(gt_class), prob)

    image_path = "{}_{}{}".format(base_path_class, "image", ext)
    cam_path = "{}_{}{}".format(base_path_class, "gradcam", ext)
    heatmap_path = "{}_{}{}".format(base_path_class, "heatmap", ext)
    segmentation_path = "{}_{}{}".format(base_path_class, "segmented", ext)

    # write images
    cv2.imwrite(image_path, image_ch3)
    cv2.imwrite(cam_path, cam)
    cv2.imwrite(heatmap_path, (heatmap * 255.0).astype(np.uint8))
    cv2.imwrite(segmentation_path, (heatmap[:, :, None].astype(float) * image_ch3).astype(np.uint8))

def plot_activations(layer_name, save_fname, sess, feed_dict):
    conv_output = sess.graph.get_tensor_by_name(layer_name + ':0')
    layers = sess.run(conv_output, feed_dict=feed_dict)
    filters = layers.shape[3]
    plt.figure(1, figsize=(15,10))
    n_columns = 24
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('f{0}'.format(str(i)))
        plt.axis('off')
        plt.grid(False)
        plt.imshow(layers[0,:,:,i], interpolation="nearest", cmap='gray')

    base_path, ext = os.path.splitext(save_fname)
    plt.savefig(base_path+'.png', dpi=200, format='png', bbox_inches='tight') 
    plt.close()

def viz_filters(layer_name,
                save_path,
                sess,
                feed_dict):

    filters = tf.get_default_graph().get_tensor_by_name(layer_name + ':0')
    learned_filters = sess.run(filters, feed_dict=feed_dict)
    learned_filters = np.array(learned_filters)[:,:,0:1,:]
    if len(learned_filters.shape) == 4:
        n_channel = learned_filters.shape[2]
    elif len(learned_filters.shape) == 3:
        n_channel = 1
        learned_filters = np.expand_dims(learned_filters, axis=2)

    filters = learned_filters.shape[3]
    plt.figure(1, figsize=(15,10))
    n_columns = 24
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('f{0}'.format(str(i)))
        plt.axis('off')
        plt.grid(False)
        plt.imshow(learned_filters[:,:,0,i], interpolation="nearest", cmap='gray')

    base_path, ext = os.path.splitext(save_path)
    plt.savefig(base_path+'.png', dpi=200, format='png', bbox_inches='tight') 
    plt.close()