#!/usr/bin/env python
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# author: Wonmin Byeon (wonmin.byeon@gmail.com)
# This code is adapted by: https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

from six.moves import urllib
from six.moves import xrange 
import tensorflow as tf

import data_reader
import grad_cam

__author__ = "Wonmin Byeon"

__maintainer__ = "Wonmin Byeon"
__email__ = "wonmin.byeon@gmail.com"

IMAGE_SIZE_W, IMAGE_SIZE_H = 180, 520
NUM_CHANNELS = 2
PIXEL_DEPTH = 255
NUM_LABELS = 2
SEED = None   # Set to None for random seed.
BATCH_SIZE = 15
NUM_EPOCHS = 1000
TEST_BATCH_SIZE = 1
EVAL_FREQUENCY = 2  # Number of steps between evaluations.

n_hidden = [32*2, 64*2, 512]
FILTER_SIZE = 5
# NUM_RUNS = 1

FLAGS = None

require_improve = 300   # for early stopping

def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32

def accuracy(predictions, labels, fname=None):
  correct_bool = (np.argmax(predictions, 1) == labels)
  if fname is not None: 
    fname1, correct_bool1 = np.array(fname), np.array(correct_bool)
  return (100.0 * np.sum(correct_bool) / predictions.shape[0])

def accuracy_perClass(predictions, labels, fname=None):
  correct_bool = (np.argmax(predictions, 1) == labels)

  correct_bool0 = ((labels == 0) & (correct_bool == True))
  correct_bool1 = ((labels == 1) & (correct_bool == True))

  if fname is not None: 
    fname1, correct_bool_ = np.array(fname), np.array(correct_bool)
  return (100.0 * np.sum(correct_bool0) / np.sum(labels == 0)), (100.0 * np.sum(correct_bool1) / np.sum(labels == 1))

def perf_measure(y_hat, y_actual):
  y_hat = np.argmax(y_hat, 1)

  TP = 0
  FP = 0
  TN = 0
  FN = 0
  for i in range(len(y_hat)): 
    if y_actual[i]==y_hat[i]==1:
       TP += 1.
    if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
       FP += 1.
    if y_actual[i]==y_hat[i]==0:
       TN += 1.
    if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
       FN += 1.

  return(TP, FP, TN, FN)

def sensitivity(y_hat, y_actual):
  TP, FP, TN, FN = perf_measure(y_hat, y_actual)
  return (TP / (TP+FN))*100.

def specificity(y_hat, y_actual):
  TP, FP, TN, FN = perf_measure(y_hat, y_actual)
  return (TN / (FP+TN))*100.

def train(curr_run, train=True, model_fname=''):
  if train:
    model_fname = "models/model_run_%d.ckpt" % int(curr_run)
  
  best_val_acc = 0.
  last_improve = 0        # Iteration-number for last improvement to validation accuracy.
  current_iter = 0        # Counter for total number of iterations performed so far.

  tf.reset_default_graph()
  train_data, test_data, train_labels, test_labels, train_fname, test_fname, min_h, min_w = data_reader.reading_data()
  if FLAGS.real_test:
  	test_data_real, test_labels_real, test_fname_real= data_reader.reading_test_data("data/realimg-test/")

  num_epochs = NUM_EPOCHS
  train_size = train_labels.shape[0]

  if not train:
    batch_size = 1
  else:
    batch_size = BATCH_SIZE
  train_data_node = tf.placeholder(data_type(), shape=(batch_size, IMAGE_SIZE_H, IMAGE_SIZE_W, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64)
  eval_data = tf.placeholder(data_type(), shape=(TEST_BATCH_SIZE, IMAGE_SIZE_H, IMAGE_SIZE_W, NUM_CHANNELS))

  conv1_weights = tf.Variable(tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, NUM_CHANNELS, n_hidden[0]],  
                          stddev=0.1, seed=SEED, dtype=data_type()), name='conv1_weights')
  conv1_biases = tf.Variable(tf.zeros([n_hidden[0]], dtype=data_type()))
  
  conv2_weights = tf.Variable(tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, n_hidden[0], n_hidden[1]], 
                              stddev=0.1, seed=SEED, dtype=data_type()), name='conv2_weights')
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[n_hidden[1]], dtype=data_type()))

  fc1_weights = tf.Variable(  
                  tf.truncated_normal([IMAGE_SIZE_H // 4 * IMAGE_SIZE_W // 4 * n_hidden[1], n_hidden[2]],
                          stddev=0.1,
                          seed=SEED,
                          dtype=data_type()))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[n_hidden[2]], dtype=data_type()))

  fco_weights = tf.Variable(tf.truncated_normal([n_hidden[2], NUM_LABELS],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()))
  fco_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=data_type()))

  def model(data, train=False):
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME', 
                        name='conv1')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME',
                          name='pool1')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        name='conv2')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME',
                          name='pool2')
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

    hidden1 = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    
    if FLAGS.do_dropout:
      hidden1 = tf.nn.dropout(hidden1, 0.8, seed=SEED)

    return tf.matmul(hidden1, fco_weights) + fco_biases

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=train_labels_node, logits=logits))

  # L2 regularization for the fully connected parameters.
  if FLAGS.do_weight_decay:
    regularizers = tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) 
    loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0, dtype=data_type())
  learning_rate = tf.train.exponential_decay(
      0.001,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.99,                # Decay rate.
      staircase=True)

  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)
  train_prediction = tf.nn.softmax(logits)

  eval_prediction = tf.nn.softmax(model(eval_data))

  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < TEST_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
    for begin in xrange(0, size, TEST_BATCH_SIZE):
      end = begin + TEST_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-TEST_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  if FLAGS.save or FLAGS.resume:  
    saver = tf.train.Saver()

  start_time = time.time()
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('Initialized!')
    # Loop through training steps.
    if train: 
      for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
        current_iter += 1

        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        
        batch_data, batch_labels, _ = data_reader.shuffling_dataset(batch_data, batch_labels, batch_labels)

        feed_dict = {train_data_node: batch_data,
                     train_labels_node: batch_labels}
        sess.run(optimizer, feed_dict=feed_dict)

        if step % EVAL_FREQUENCY == 0:
          l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                        feed_dict=feed_dict)
          elapsed_time = time.time() - start_time
          start_time = time.time()

          train_accuracy = accuracy(predictions, batch_labels)
          test_pred = eval_in_batches(test_data, sess)
          test_accuracy = accuracy(test_pred, test_labels)#, test_fname)
          TP, FP, TN, FN = perf_measure(test_pred, test_labels)
          sens = sensitivity(test_pred, test_labels)
          spec = specificity(test_pred, test_labels)
          if FLAGS.real_test:        
            test_pred_real = eval_in_batches(test_data_real, sess)
            test_accuracy_real = accuracy(test_pred_real, test_labels_real)#, test_fname)

          print(curr_run, 'Runs, Step %d (epoch %.2f), %.1f ms' 
                                % (step, float(step) * BATCH_SIZE / train_size, 
                                  1000 * elapsed_time / EVAL_FREQUENCY))
          print('Minibatch loss: %.3f, learning rate: %.6f, last improve/current_iter: %d/%d, %d' 
                                % (l, lr, last_improve, current_iter, current_iter - last_improve))
          print('Minibatch accuracy: %.1f%%' % train_accuracy)
          print('test accuracy: %.1f%%' % test_accuracy)
          print('sensitivity:', sens, 'specificity:', spec)
          if FLAGS.real_test:
            print('real data test accuracy: %.1f%%' % test_accuracy_real)
          print('------------------------------------')
          sys.stdout.flush()

          if train_accuracy > best_val_acc:
            best_val_acc = train_accuracy
            last_improve = current_iter
            if FLAGS.save:
              save_path = saver.save(sess, model_fname)
              print("Model saved: %s" % save_path)

          if current_iter - last_improve > require_improve:
            print("No improvement found in a while, stopping optimization.")
            break
    if FLAGS.resume:
      print("Model restored: %s" % model_fname)
      saver.restore(sess, model_fname)

    # Finally print the result!
    test_pred = eval_in_batches(test_data, sess)
    test_accuracy = accuracy(test_pred, test_labels)
    test_accuracy_cl0, test_accuracy_cl1  = accuracy_perClass(test_pred, test_labels, test_fname)
    sens = sensitivity(test_pred, test_labels)
    spec = specificity(test_pred, test_labels)
    if FLAGS.real_test:
      test_pred_real = eval_in_batches(test_data_real, sess)
      test_accuracy_real = accuracy(test_pred_real, test_labels_real)

    print('Test accuracy: %.1f%%' % test_accuracy, test_accuracy_cl0, test_accuracy_cl1)
    print('sensitivity:', sens, 'specificity:', spec)
    if FLAGS.real_test:
      print('real data test accuracy: %.1f%%' % test_accuracy_real)
    

    if FLAGS.vis:
      type='train'
      gt_labels=test_labels
      if type == 'train':
        gt_labels=train_labels
        test_pred = eval_in_batches(train_data, sess)
        test_accuracy = accuracy(test_pred, test_labels)
        test_accuracy_cl0, test_accuracy_cl1  = accuracy_perClass(test_pred, train_labels, train_fname)
        sens = sensitivity(test_pred, train_labels)
        spec = specificity(test_pred, train_labels)
        correct_bool = (np.argmax(test_pred, 1) == train_labels)
        fname1 = np.array(train_fname)
        print('Train accuracy: %.1f%%' % test_accuracy, test_accuracy_cl0, test_accuracy_cl1)

      model_fname_ = model_fname.split('/')[-1]
      pred_labels = np.argmax(test_pred, 1)
      for i, prob_set in enumerate(test_pred):
        if type == 'train':
          feed_dict = {train_data_node: train_data[i:i+1, ...],
                       train_labels_node: train_labels[i:i+1]}
          input_image = train_data[i:i+1, ...]
        elif type == 'test':
          feed_dict = {train_data_node: test_data[i:i+1, ...], 
                      train_labels_node: test_labels[i:i+1]}
          input_image = test_data[i:i+1, ...]
        else:
          raise NotImplementedError
        
        pred_class = pred_labels[i]
        gt_class = gt_labels[i]
        fname = fname1[i].replace('/','_')
        prob = prob_set[pred_class]
        ##### GRAD-CAM (Gradient-based Localization)
        if FLAGS.vis == 'gradcam':
          if pred_class == gt_class:
            save_fname = 'vis-grad-cam/'+type+"/"+model_fname_+"/correct/"+fname
          else:
            save_fname = 'vis-grad-cam/'+type+"/"+model_fname_+"/wrong/"+fname
          cams = grad_cam.grad_cam(loss, "conv2", sess, feed_dict=feed_dict)
          print(i, pred_class, gt_class, prob, save_fname)
          grad_cam.save_cam(cams, i, pred_class, gt_class, prob, input_image, save_fname)

        ##### Visualize activations
        elif FLAGS.vis == 'activations':
          layers = ["conv1" ,"conv2"]
          # Plot out the Layers
          for layer in layers:
            if pred_class == gt_class:
              save_fname = 'vis-activation/'+type+"/"+model_fname_+"/correct/"+layer+"_"+fname
            else:
              save_fname = 'vis-activation/'+type+"/"+model_fname_+"/wrong/"+layer+"_"+fname
            grad_cam.plot_activations(layer, save_fname, sess, feed_dict=feed_dict)

        ##### Visualize Filters
        elif FLAGS.vis == 'filters':
          layers = [('conv1_weights', conv1_weights),('conv1_weights', conv2_weights)]
          # Plot out the Layers
          for name, layer in layers:
            if pred_class == gt_class:
              save_fname = 'vis-filters/'+type+"/"+model_fname_+"/correct/"+name+"_"+fname
            else:
              save_fname = 'vis-filters/'+type+"/"+model_fname_+"/wrong/"+name+"_"+fname
            grad_cam.viz_filters(name, save_fname, sess, feed_dict=feed_dict)
        else:
          raise NotImplementedError

  return test_accuracy, [test_accuracy_cl0, test_accuracy_cl1], [sens, spec]


test_acc_set, test_acc_set_cl0, test_acc_set_cl1 = [], [], []
sens_set, spec_set = [], []

tf.reset_default_graph()
parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--save', dest='save', action='store_true')
parser.add_argument('--resume', dest='resume', action='store_true')
parser.add_argument('--real_test', dest='real_test', action='store_true')
parser.add_argument(
    '--use_fp16',
    default=False,
    help='Use half floats instead of full floats if True.',
    action='store_true')
parser.add_argument(
    '--do_dropout',
    default=False,
    help='True if adding dropout.')
parser.add_argument(
    '--do_weight_decay',
    default=False,
    help='True if adding weight_decay.')
parser.add_argument(
    '--num_runs',
    type=int,
    default=1,  
    help='batch size')
parser.add_argument(
    '--model_fname',
    type=str,
    default='',  
    help='model filename')
parser.add_argument(
    '--vis',
    type=str,
    default=None,  
    help='model filename')

FLAGS = parser.parse_args()

for rr in xrange(FLAGS.num_runs):  
  test_accuracy, test_accuracy_cl, measure = train(rr+1, FLAGS.train, FLAGS.model_fname)

  test_acc_set.append(test_accuracy)
  test_acc_set_cl0.append(test_accuracy_cl[0])
  test_acc_set_cl1.append(test_accuracy_cl[1])
  sens_set.append(measure[0]) 
  spec_set.append(measure[1])

  print(rr+1, "runs: mean",np.mean(test_acc_set), "std",np.std(test_acc_set))
  print("         for class 0",np.mean(test_acc_set_cl0), "std",np.std(test_acc_set_cl0))
  print("         for class 1",np.mean(test_acc_set_cl1), "std",np.std(test_acc_set_cl1))
  print("         sensitivity",np.mean(sens_set), "std",np.std(sens_set))
  print("         specificity",np.mean(spec_set), "std",np.std(spec_set))
