#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 02:29:50 2018

@author: loktar
"""

from __future__ import print_function
import sys
import argparse
import os
sys.dont_write_bytecode = True
import pandas as pd
import numpy as np

import tensorflow as tf
from data_io import data_loader
from resnet import get_resnet
from losses import get_loss
from optimizer import get_optimizer, get_decay_category
from metrics import categorical_accuracy
from visualization import TensorBoard

from config import config_data_loader, config_valid_loader, config_common, config_trainer

class base_trainer(object):

    def __init__(self, args):

        self.input_shape        = config_common['input_shape']
        self.num_classes        = config_common['num_classes']

        self.model_name         = config_trainer['model_name']
        self.num_epochs         = config_trainer['num_epochs']
        self.steps_per_epoch    = config_trainer['step_per_epoch']
        self.device             = config_trainer['device']

        self.save_path          = config_trainer['save_path']
        self.save_steps         = config_trainer['save_steps']
        self.restore_path       = os.path.join(self.save_path, 'checkpoints')
        self.checkpoints_path   = os.path.join(self.save_path, 'checkpoints/cifar10')
        self.log_path           = os.path.join(self.save_path, 'train_log.csv')


        self.item_name_lst = ['epoch','global_step','loss', 'accuracy']

        self.optimizer_name = args.optimizer
        self.start_lr = args.start_lr

        self.l2_weight = 2e-04

        self.global_step = 0
        self.epoch = 0

        self.best_accuracy = 0
        self.best_epoch = 0

    def _load_data(self):
        """ loading data
        """

        with tf.variable_scope('DataLoader'):
            self.train_iterator = data_loader(**config_data_loader)
            self.train_loader = self.train_iterator.get_next()

            self.valid_iterator = data_loader(**config_valid_loader)
            self.valid_loader = self.valid_iterator.get_next()


    def _set_placeholder(self):
        """ seting placeholder
        """
        with tf.variable_scope("placeholder"):
            self.input = tf.placeholder(dtype=tf.float32, shape=(None,)+self.input_shape, name='input')
            self.label = tf.placeholder(dtype=tf.float32, name='label')

    def _load_model(self, network=None):
        """ loading model
        """
        with tf.variable_scope("model_network"):
            self.net = get_resnet(self.model_name, num_classes=self.num_classes)(self.input)
            self.net.build_model()
            self.pred = self.net.outputs

    def _set_loss(self):
        """ set loss module
        """
        with tf.variable_scope("loss"):
            self.SCE_loss = get_loss('softmax_cross_entropy')(self.label, self.pred)
            l2_var_lst = [i for i in tf.trainable_variables() if "kernel" in i.name]
            self.l2_loss = get_loss('l2_loss')(l2_var_lst)
            self.loss = self.SCE_loss + self.l2_weight * self.l2_loss

    def _set_metrics(self):
        """ set metrics module
        """
        with tf.variable_scope("metrics"):
            self.accuracy = categorical_accuracy(self.label, self.pred)
            self.metrics_lst = [self.accuracy]

    def _set_optimizer(self):
        """ set optimizer module
        """
        with tf.variable_scope("optimizer"):
            # self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model_network/Classifier")
            self.tvars = tf.trainable_variables()
            print("model trainable variables:")
            print("================================")
            for i in self.tvars:
                print(i.name)
            self.grads = tf.gradients(self.loss, self.tvars)
            self.lr = get_decay_category('exponential')(self.start_lr, self.epoch, 5, 0.5)
            # self.lr = tf.placeholder_with_default(self.start_lr, shape=None, name="lr_placeholder")
            self.optimizer = get_optimizer(self.optimizer_name)(self.lr)
            self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.tvars))
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def _set_tensorboard(self):
        """ set tensorboard module
        """
        with tf.variable_scope("model_summary"):
            with tf.device('/cpu:0'):
                summary_tool = TensorBoard()
                summary_tool.scalar_summary("all_loss", self.loss)
                summary_tool.scalar_summary("SCE_loss", self.SCE_loss)
                summary_tool.scalar_summary("l2_loss", self.l2_loss)
                summary_tool.scalar_summary("accuracy", self.accuracy)
                summary_tool.scalar_summary("learning_rate", self.lr)
                summary_tool.hist_summary("features", self.net.features)
                summary_tool.hist_summary("outputs", self.net.outputs)
                self.summary_ops = summary_tool.merge_all_summary()

    def _set_saver(self):
        """set saver module
        """
        with tf.variable_scope("saver"):
            self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=100)
            self.writer = tf.summary.FileWriter(os.path.join(self.save_path, 'train_summary'), self.sess.graph)

    def _set_logs(self):
        """set logs module
        """
        self.logs = dict()
        for item_name in self.item_name_lst:
            self.logs[item_name] = []

    def _set_session(self):
        """ set session module
        """
        with tf.variable_scope("Session"):
            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True
            # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.3
            self.sess = tf.Session(config=sess_config)

    def _init_or_restore(self):
        """ initialize model or restore from checkpoint
        """
        with tf.variable_scope("initialization"):
            model_file = tf.train.latest_checkpoint(self.restore_path)
            try:
                self.sess.run(tf.global_variables_initializer())
                self.saver.restore(self.sess, model_file)
                print('Restore Sucessful!')

            except:
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(tf.local_variables_initializer())
                print('Restore Failed!')

    def build(self):
        """ build trainer
        """
        with tf.device('/cpu:0'):
            self._load_data()
        with tf.device('/{}'.format(self.device)):
            self._set_placeholder()
            self._load_model()
            self._set_loss()
            self._set_metrics()
            self._set_optimizer()

        self._set_tensorboard()
        self._set_session()
        self._set_saver()
        self._set_logs()
        self._init_or_restore()

    def train_step(self):
        """ one step training loop
        """
        try:
            image_batch, label_batch = self.sess.run(self.train_loader)
        except tf.errors.OutOfRangeError:
            self.sess.run(self.train_iterator.initializer)
            image_batch, label_batch = self.sess.run(self.train_loader)

        train_feed = {self.input: image_batch,
                      self.label: label_batch,
                      self.net.training: True,}

        run_lst = [self.train_op, self.extra_update_ops, self.summary_ops, self.loss, self.accuracy]
        _, _, summary_str, loss, accuracy = self.sess.run(run_lst, feed_dict=train_feed)
        print("[epoch {}/{} global_step {}] loss={:.4f} accuracy={:.4f}".format(self.epoch+1, self.num_epochs, self.global_step, loss, accuracy))

        self.logs['epoch'].append(self.epoch)
        self.logs['global_step'].append(self.global_step)
        self.logs['loss'].append(loss)
        self.logs['accuracy'].append(accuracy)

        if (self.global_step + 1) % self.save_steps == 0:
            log_df = pd.DataFrame(self.logs)
            log_df.to_csv(self.log_path, index=None)
            self.saver.save(self.sess, self.checkpoints_path, self.global_step)

        self.writer.add_summary(summary_str, self.global_step)
        self.global_step += 1

    def train_loop(self):
        """ one epoch training loop
        """
        self.sess.run(self.train_iterator.initializer)
        self.sess.run(self.valid_iterator.initializer)
        for self.step in range(self.steps_per_epoch):
            self.train_step()

    def valid_process(self):
        """ validation process
        """
        self.sess.run(self.valid_iterator.initializer)
        loss_lst = []
        accuracy_lst = []

        while True:
            try:
                image_batch, label_batch = self.sess.run(self.valid_loader)
            except tf.errors.OutOfRangeError:
                break

            valid_feed = {self.input: image_batch,
                          self.label: label_batch,
                          self.net.training: False}

            run_lst = [self.loss, self.accuracy]
            loss, accuracy = self.sess.run(run_lst, feed_dict=valid_feed)
            loss_lst.append(loss)
            accuracy_lst.append(accuracy)

        loss_mean = np.mean(loss_lst)
        acc_mean = np.mean(accuracy_lst)

        print("[epoch: {}] valid_loss: {}, valid_acc: {}".format(self.epoch, loss_mean, acc_mean))
        if acc_mean > self.best_accuracy:
            print("validation loss has been increased from {:.6f} to {:.6f}.".format(self.best_accuracy, acc_mean))
            self.best_accuracy = acc_mean
            self.best_epoch = self.epoch
        else:
            print("validation loss is not increased.")
        print("The best epoch is epoch-{}, and the best accuracy is {}".format(self.best_epoch, self.best_accuracy))


    def run(self):
        """ deal with Train Runner
        """
        for self.epoch in range(self.num_epochs):
            self.train_loop()
            self.valid_process()



def parameters_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--start_lr", type=float, default=0.0001)

    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parameters_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    trainer = base_trainer(args)
    trainer.build()
    trainer.run()
