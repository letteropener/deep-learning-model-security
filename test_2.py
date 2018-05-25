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
from cleverhans.attacks import SaliencyMapMethod, FastGradientMethod, BasicIterativeMethod, ElasticNetMethod, DeepFool, VirtualAdversarialMethod
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils import AccuracyReport, set_log_level

import os

FLAGS = flags.FLAGS


def JSMA_FGSM(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64):
    """
    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param clean_train: if true, train on clean examples
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    sess = tf.Session()

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)
    source_samples = batch_size
    # Use label smoothing
    # Hopefully this doesn't screw up JSMA...
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    model_path = "models/mnist"
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    eval_par = {'batch_size': batch_size}
    rng = np.random.RandomState([2017, 8, 30])

    if clean_train:
        model = make_basic_cnn(nb_filters=nb_filters)
        preds = model.get_probs(x)

        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_test, Y_test, args=eval_params)
            report.clean_train_clean_eval = acc
            assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('Test accuracy on legitimate examples: %0.4f' % acc)
        model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                    args=train_params, rng=rng)
        print("#####Starting attacks on clean model#####")
        #################################################################
        #Clean test against JSMA
        jsma_params = {'theta': 1., 'gamma': 0.1,
               'clip_min': 0., 'clip_max': 1.,
               'y_target': None}

        jsma = SaliencyMapMethod(model, back='tf', sess=sess)
        adv_x = jsma.generate(x, **jsma_params)
        preds_adv = model.get_probs(adv_x)

        # Evaluate the accuracy of the MNIST model on FGSM adversarial examples
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Clean test accuracy on JSMA adversarial examples: %0.4f' % acc)
        ################################################################
        #Clean test against FGSM
        fgsm_params = {'eps': 0.3,
                       'clip_min': 0.,
                       'clip_max': 1.}

        fgsm = FastGradientMethod(model, sess=sess)
        adv_x = fgsm.generate(x, **fgsm_params)
        preds_adv = model.get_probs(adv_x)

        # Evaluate the accuracy of the MNIST model on FGSM adversarial examples
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Clean test accuracy on FGSM adversarial examples: %0.4f' % acc)
        ################################################################
        #Clean test against BIM
        bim_params = {'eps': 0.3,
                      'eps_iter': 0.01,
                      'nb_iter': 100,
                      'clip_min': 0.,
                      'clip_max': 1.}
        bim = BasicIterativeMethod(model, sess=sess)
        adv_x = bim.generate(x, **bim_params)
        preds_adv = model.get_probs(adv_x)
        
        # Evaluate the accuracy of the MNIST model on FGSM adversarial examples
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Clean test accuracy on BIM adversarial examples: %0.4f' % acc)
        ################################################################
        #Clean test against EN
        en_params = {'binary_search_steps': 1,
             #'y': None,
             'max_iterations': 100,
             'learning_rate': 0.1,
             'batch_size': source_samples,
             'initial_const': 10}
        en = ElasticNetMethod(model, back='tf', sess=sess)
        adv_x = en.generate(x, **en_params)
        preds_adv = model.get_probs(adv_x)
        
        # Evaluate the accuracy of the MNIST model on FGSM adversarial examples
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Clean test accuracy on EN adversarial examples: %0.4f' % acc)
        ################################################################
        #Clean test against DF
        deepfool_params = {'nb_candidate':10,
                           'overshoot':0.02,
                           'max_iter': 50,
                           'clip_min': 0.,
                           'clip_max': 1.}
        deepfool = DeepFool(model, sess=sess)
        adv_x = deepfool.generate(x, **deepfool_params)
        preds_adv = model.get_probs(adv_x)
        
        # Evaluate the accuracy of the MNIST model on FGSM adversarial examples
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Clean test accuracy on DF adversarial examples: %0.4f' % acc)
        ################################################################
        #Clean test against VAT
        vat_params = {'eps': 2.0,
                      'num_iterations': 1,
                      'xi': 1e-6,
                      'clip_min': 0.,
                      'clip_max': 1.}
        vat = VirtualAdversarialMethod(model, sess=sess)
        adv_x = vat.generate(x, **vat_params)
        preds_adv = model.get_probs(adv_x)
        
        # Evaluate the accuracy of the MNIST model on FGSM adversarial examples
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Clean test accuracy on VAT adversarial examples: %0.4f\n' % acc)
        ################################################################
        print("Repeating the process, using adversarial training\n")
    # Redefine TF model graph
    model_2 = make_basic_cnn(nb_filters=nb_filters)
    preds_2 = model_2(x)
    #################################################################
    #Adversarial test against JSMA
    jsma_params = {'theta': 1., 'gamma': 0.1,
           'clip_min': 0., 'clip_max': 1.,
           'y_target': None}

    jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    adv_x = jsma.generate(x, **jsma_params)
    preds_adv_jsma = model.get_probs(adv_x)
    ################################################################
    #Adversarial test against FGSM
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}

    fgsm = FastGradientMethod(model, sess=sess)
    adv_x = fgsm.generate(x, **fgsm_params)
    preds_adv_fgsm = model.get_probs(adv_x)
    ################################################################
    #Adversarial test against BIM
    bim_params = {'eps': 0.3,
                  'eps_iter': 0.01,
                  'nb_iter': 100,
                  'clip_min': 0.,
                  'clip_max': 1.}
    bim = BasicIterativeMethod(model, sess=sess)
    adv_x = bim.generate(x, **bim_params)
    preds_adv_bim = model.get_probs(adv_x)
    ################################################################
    #Adversarial test against EN
    en_params = {'binary_search_steps': 5,
         #'y': None,
         'max_iterations': 100,
         'learning_rate': 0.1,
         'batch_size': source_samples,
         'initial_const': 10}
    en = ElasticNetMethod(model, back='tf', sess=sess)
    adv_x = en.generate(x, **en_params)
    preds_adv_en = model.get_probs(adv_x)
    ################################################################
    #Adversarial test against DF
    deepfool_params = {'nb_candidate':10,
                       'overshoot':0.02,
                       'max_iter': 200,
                       'clip_min': 0.,
                       'clip_max': 1.}
    deepfool = DeepFool(model, sess=sess)
    adv_x = deepfool.generate(x, **deepfool_params)
    preds_adv_df = model.get_probs(adv_x)
    ################################################################
    #Adversarial test against VAT
    vat_params = {'eps': 2.0,
                  'num_iterations': 1,
                  'xi': 1e-6,
                  'clip_min': 0.,
                  'clip_max': 1.}
    vat = VirtualAdversarialMethod(model, sess=sess)
    adv_x = vat.generate(x, **vat_params)
    preds_adv_vat = model.get_probs(adv_x)
    ################################################################
    print("#####Evaluate trained model#####")
    def evaluate_2():
        # Evaluate the accuracy of the MNIST model on JSMA adversarial examples
        acc = model_eval(sess, x, y, preds_adv_jsma, X_test, Y_test, args=eval_par)
        print('Test accuracy on JSMA adversarial examples: %0.4f' % acc)

        # Evaluate the accuracy of the MNIST model on FGSM adversarial examples
        acc = model_eval(sess, x, y, preds_adv_fgsm, X_test, Y_test, args=eval_par)
        print('Test accuracy on FGSM adversarial examples: %0.4f' % acc)

        # Evaluate the accuracy of the MNIST model on BIM adversarial examples
        acc = model_eval(sess, x, y, preds_adv_bim, X_test, Y_test, args=eval_par)
        print('Test accuracy on BIM adversarial examples: %0.4f' % acc)

        # Evaluate the accuracy of the MNIST model on EN adversarial examples
        acc = model_eval(sess, x, y, preds_adv_en, X_test, Y_test, args=eval_par)
        print('Test accuracy on EN adversarial examples: %0.4f' % acc)

        # Evaluate the accuracy of the MNIST model on DF adversarial examples
        acc = model_eval(sess, x, y, preds_adv_df, X_test, Y_test, args=eval_par)
        print('Test accuracy on DF adversarial examples: %0.4f' % acc)

        # Evaluate the accuracy of the MNIST model on VAT adversarial examples
        acc = model_eval(sess, x, y, preds_adv_vat, X_test, Y_test, args=eval_par)
        print('Test accuracy on VAT adversarial examples: %0.4f\n' % acc)
        
    preds_2_adv = [preds_adv_jsma 
                    ,preds_adv_fgsm 
                   # ,preds_adv_bim 
                   # ,preds_adv_en 
                   # ,preds_adv_df
                   ]

    model_train(sess, x, y, preds_2, X_train, Y_train,
                predictions_adv=preds_2_adv,evaluate = evaluate_2,
                args=train_params, rng=rng)

   



