"""
This
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod, ElasticNetMethod, CarliniWagnerL2
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils import AccuracyReport, set_log_level
from test_1 import JSMA
from test_2 import JSMA_FGSM
from test_3 import JSMA_FGSM_BIM
from test_4 import JSMA_FGSM_BIM_EN
from test_5 import JSMA_FGSM_BIM_EN_DF
from test_6 import JSMA_FGSM_BIM_EN_DF_VAT
import os

FLAGS = flags.FLAGS
"""
My order:
1. JSMA
2. JSMA FGSM
3. JSMA FGSM BIM
4. JSMA FGSM BIM EN
5. JSMA FGSM BIM EN DeepFool
6. JSMA FGSM BIM EN DeepFool VAT
Epoch: 6
Batch size: 1000
Train: 6000
Test: 1000

Test with clean model
Train with adversarial examples
Evaluate trained model
"""
def main(argv=None):
    
    print("***********************************")
    print("**********Starting Test 1**********")
    print("***********************************")
    JSMA(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters,
                   train_start=FLAGS.train_start,
                   train_end=FLAGS.train_end,
                   test_start=FLAGS.test_start,
                   test_end=FLAGS.test_end)
    print("***********************************")
    print("**********Starting Test 2**********")
    print("***********************************")
    JSMA_FGSM(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters,
                   train_start=FLAGS.train_start,
                   train_end=FLAGS.train_end,
                   test_start=FLAGS.test_start,
                   test_end=FLAGS.test_end)       
    print("***********************************")
    print("**********Starting Test 3**********")
    print("***********************************")          
    JSMA_FGSM_BIM(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters,
                   train_start=FLAGS.train_start,
                   train_end=FLAGS.train_end,
                   test_start=FLAGS.test_start,
                   test_end=FLAGS.test_end)
    print("***********************************")
    print("**********Starting Test 4**********")
    print("***********************************")
    JSMA_FGSM_BIM_EN(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters,
                   train_start=FLAGS.train_start,
                   train_end=FLAGS.train_end,
                   test_start=FLAGS.test_start,
                   test_end=FLAGS.test_end)
    print("***********************************")
    print("**********Starting Test 5**********")
    print("***********************************")
    JSMA_FGSM_BIM_EN_DF(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters,
                   train_start=FLAGS.train_start,
                   train_end=FLAGS.train_end,
                   test_start=FLAGS.test_start,
                   test_end=FLAGS.test_end)
    
    print("***********************************")
    print("**********Starting Test 6**********")
    print("***********************************")
    JSMA_FGSM_BIM_EN_DF_VAT(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters,
                   train_start=FLAGS.train_start,
                   train_end=FLAGS.train_end,
                   test_start=FLAGS.test_start,
                   test_end=FLAGS.test_end)
    
                   
if __name__ == '__main__':    
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 10, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))
    flags.DEFINE_integer('train_start', 0, 'start of MNIST training samples')
    flags.DEFINE_integer('train_end', 10, 'end of MNIST training samples')
    flags.DEFINE_integer('test_start', 0, 'start of MNIST test samples')
    flags.DEFINE_integer('test_end', 10, 'end of MNIST test samples')


    tf.app.run()
