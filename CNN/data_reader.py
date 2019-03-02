#!/usr/bin/env python

# author: Wonmin Byeon (wonmin.byeon@gmail.com)
# data reader

import scipy.misc
import numpy as np
import glob
import random
import cv2

__author__ = "Wonmin Byeon"

__maintainer__ = "Wonmin Byeon"
__email__ = "wonmin.byeon@gmail.com"

np.random.seed(1234)

WORK_DIRECTORY = 'data/'
NUM_TEST_TM, NUM_TEST_RCM, NUM_TEST_CM = 10, 8, 2
IMAGE_SIZE_W, IMAGE_SIZE_H = 180, 520

def resize_image(image):
    return scipy.misc.imresize(image, (IMAGE_SIZE_H, IMAGE_SIZE_W))

def resize_images(data, n_data):
    resized_data = []
    for idx in xrange(n_data):
      image = data[idx]
      resized_data.append(resize_image(image))

    print("resized_data shape", np.array(resized_data).shape)
    return resized_data

def apply_clahe(image):
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  return clahe.apply(image).reshape(IMAGE_SIZE_H, IMAGE_SIZE_W, 1)

def apply_histogrameq(image):
  processed = cv2.equalizeHist(image).reshape(IMAGE_SIZE_H, IMAGE_SIZE_W, 1)
  return processed


def normalization(train_data, test_data):
  print("before norm", np.mean(train_data), np.std(train_data), np.mean(test_data), np.std(test_data))
  mean, std = np.mean(train_data), np.std(train_data)
  train_data -= mean # zero-center
  test_data -= mean
  train_data /= std
  test_data /= std
  print("after norm", np.mean(train_data), np.std(train_data), np.mean(test_data), np.std(test_data))

  return train_data, test_data


def reading_data():
  # Get the data.
  try:
      flist_tm, flist_rcm, flist_cm = glob.glob(WORK_DIRECTORY+"tm/*.jpg"), glob.glob(WORK_DIRECTORY+"rcm/*.jpg"), glob.glob(WORK_DIRECTORY+"cm/*.jpg")
      print('num files tm/rcm/cm: ',len(flist_tm), len(flist_rcm), len(flist_cm))
  except:
      print('Please set the correct path to the dataset: '+WORK_DIRECTORY+'*.jpg',)
      sys.exit()

  flist_tm, flist_rcm, flist_cm = shuffling(flist_tm), shuffling(flist_rcm), shuffling(flist_cm)
  min_w, min_h = 99999, 99999
  train_data, test_data, train_labels, test_labels, train_fname, test_fname = [], [], [], [], [], []
  count = 0

  for idx, fname in enumerate(flist_tm):
    if ".jpg" in fname:
      # image = misc.imread(fname)
      image = scipy.misc.imread(fname)
      hh, ww = image.shape
      image = resize_image(image).reshape(IMAGE_SIZE_H, IMAGE_SIZE_W, 1)
      processed1 = apply_histogrameq(image)
      image = np.concatenate([image, processed1], axis=2)

      if count < NUM_TEST_TM:
        test_data.append(image/255.)
        test_labels.append(0)
        test_fname.append(fname)
        count += 1
      else:
        train_data.append(image/255.)
        train_labels.append(0)
        train_fname.append(fname)
        
      min_w, min_h = np.amin([min_w, ww]), np.amin([min_h, hh])
      max_w, max_h = np.amax([min_w, ww]), np.amax([min_h, hh])


  count=0
  for idx, fname in enumerate(flist_rcm):
    if ".jpg" in fname:
      image = scipy.misc.imread(fname)
      hh, ww = image.shape
      image = resize_image(image).reshape(IMAGE_SIZE_H, IMAGE_SIZE_W, 1)
      processed1 = apply_histogrameq(image)
      # processed2 = apply_clahe(image)
      image = np.concatenate([image, processed1], axis=2)
      if count < NUM_TEST_RCM:
        test_data.append(image/255.)
        test_labels.append(1)
        test_fname.append(fname)
        count += 1
      else:
        train_data.append(image/255.)
        train_labels.append(1)
        train_fname.append(fname)
        
      min_w, min_h = np.amin([min_w, ww]), np.amin([min_h, hh])
      max_w, max_h = np.amax([min_w, ww]), np.amax([min_h, hh])

  count=0
  for idx, fname in enumerate(flist_cm):
    if ".jpg" in fname:
      image = scipy.misc.imread(fname)
      hh, ww = image.shape
      image = resize_image(image).reshape(IMAGE_SIZE_H, IMAGE_SIZE_W, 1)
      processed1 = apply_histogrameq(image)
      # processed2 = apply_clahe(image)
      image = np.concatenate([image, processed1], axis=2)
      if count < NUM_TEST_CM:
        test_data.append(image/255.)
        test_labels.append(1)
        test_fname.append(fname)
        count += 1
      else:
        train_data.append(image/255.)
        train_labels.append(1)
        train_fname.append(fname)
        
      min_w, min_h = np.amin([min_w, ww]), np.amin([min_h, hh])
      max_w, max_h = np.amax([min_w, ww]), np.amax([min_h, hh])

  train_data, train_labels, train_fname = shuffling_dataset(train_data, train_labels, train_fname)
  test_data, test_labels, test_fname = shuffling_dataset(test_data, test_labels, test_fname)

  train_data, test_data = np.float32(train_data), np.float32(test_data)
  train_labels, test_labels = np.int64(train_labels), np.int64(test_labels)

  return train_data, test_data, train_labels, test_labels, train_fname, test_fname, min_h, min_w

def reading_test_data(directory):
  # Get the data.
  try:
      flist_tm, flist_cm = glob.glob(directory+"tm/*.jpg"), glob.glob(directory+"cm/*.jpg")
  except:
      print('Please set the correct path to the dataset: '+directory+'*.jpg',)
      sys.exit()

  flist_tm, flist_cm = shuffling(flist_tm), shuffling(flist_cm)
  test_labels, test_data, test_fname = [], [], []
  count = 0

  for idx, fname in enumerate(flist_tm):
    if ".jpg" in fname:
      image = scipy.misc.imread(fname)
      hh, ww = image.shape
      image = resize_image(image).reshape(IMAGE_SIZE_H, IMAGE_SIZE_W, 1)
      processed1 = apply_histogrameq(image)
      image = np.concatenate([image, processed1], axis=2)
      test_data.append(image/255.)
      test_labels.append(0)
      test_fname.append(fname)
      count += 1

  count=0
  for idx, fname in enumerate(flist_cm):
    if ".jpg" in fname:
      image = scipy.misc.imread(fname)
      hh, ww = image.shape
      image = resize_image(image).reshape(IMAGE_SIZE_H, IMAGE_SIZE_W, 1)
      processed1 = apply_histogrameq(image)

      image = np.concatenate([image, processed1], axis=2)
      test_data.append(image/255.)
      test_labels.append(1)
      test_fname.append(fname)
      count += 1
        
  test_data, test_labels, test_fname = shuffling_dataset(test_data, test_labels, test_fname)

  test_data = np.float32(test_data)
  test_labels = np.int64(test_labels)

  return test_data, test_labels, test_fname

def shuffling(data):
  perm = np.arange(len(data))
  np.random.shuffle(perm)
  data = np.array(data)
  return data[perm]

def shuffling_dataset(data, labels, fname):
  perm = np.arange(len(data))
  np.random.shuffle(perm)
  data = np.array(data)
  labels = np.array(labels)
  fname = np.array(fname)
  return data[perm], labels[perm], fname[perm]
