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
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils import AccuracyReport, set_log_level

import os

FLAGS = flags.FLAGS


def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
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

    # Use label smoothing
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
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    bim_params = {'eps': 0.3,
                  'eps_iter': 0.01,
                  'nb_iter': 100,
                  'clip_min': 0.,
                  'clip_max': 1.}
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

        # Calculate training error
        if testing:
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_train, Y_train, args=eval_params)
            report.train_clean_train_clean_eval = acc

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        fgsm = FastGradientMethod(model, sess=sess)
        adv_x = fgsm.generate(x, **fgsm_params)
        preds_adv = model.get_probs(adv_x)

        # Evaluate the accuracy of the MNIST model on FGSM adversarial examples
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Test accuracy on FGSM adversarial examples: %0.4f\n' % acc)

        # Calculate training error
        if testing:
            eval_par = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds_adv, X_train,
                             Y_train, args=eval_par)
            report.train_clean_train_adv_eval = acc

        # Init the Basic Iterative Method attack object and graph
        bim = BasicIterativeMethod(model, sess=sess)
        adv_x_2 = bim.generate(x, **bim_params)
        preds_adv_2 = model.get_probs(adv_x_2)

        # Evaluate the accuracy of the MNIST model on Basic Iterative Method adversarial examples
        acc = model_eval(sess, x, y, preds_adv_2, X_test, Y_test, args=eval_par)
        print('Test accuracy on Basic Iterative Method adversarial examples: %0.4f\n' % acc)

        # Calculate training error
        if testing:
            eval_par = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds_adv, X_train,
                             Y_train, args=eval_par)
            report.train_clean_train_adv_eval = acc

        print("Repeating the process, using adversarial training")
    # Redefine TF model graph
    model_2 = make_basic_cnn(nb_filters=nb_filters)
    preds_2 = model_2(x)
    fgsm2 = FastGradientMethod(model_2, sess=sess)
    adv_x_fgsm = fgsm2.generate(x, **fgsm_params)
    if not backprop_through_attack:
        # For the fgsm attack used in this tutorial, the attack has zero
        # gradient so enabling this flag does not change the gradient.
        # For some other attacks, enabling this flag increases the cost of
        # training, but gives the defender the ability to anticipate how
        # the atacker will change their strategy in response to updates to
        # the defender's parameters.
        adv_x_fgsm = tf.stop_gradient(adv_x_fgsm)
    preds_2_adv_fgsm = model_2(adv_x_fgsm)

    bim2 = BasicIterativeMethod(model_2, sess=sess)
    adv_x_bim = bim2.generate(x, **bim_params)
    preds_2_adv_bim = model_2(adv_x_bim)

    # X_train_rep = np.tile(X_train, [2, 1, 1, 1])
    # Y_train_rep = np.tile(Y_train, [2, 1])
    # preds_2_rep = tf.tile(preds_2, [2, 1])
    # preds_2_adv = model_2(tf.concat([adv_x_fgsm, adv_x_bim], 0))
    #
    # # Perform and evaluate adversarial training
    # model_train(sess, x, y, preds_2_rep, X_train_rep, Y_train_rep,
    #             predictions_adv=preds_2_adv,
    #             args=train_params, rng=rng)

    def evaluate_2():
        # evaluate the final result of the model
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)

        # Accuracy of the adversarially trained model on FGSM adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv_fgsm, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on FGSM adversarial examples: %0.4f' % accuracy)

        # Accuracy of the adversarially trained model on Basic Iterative Method adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv_bim, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on BIM adversarial examples: %0.4f' % accuracy)

    preds_2_adv = [preds_2_adv_fgsm, preds_2_adv_bim]
    model_train(sess, x, y, preds_2, X_train, Y_train,
                predictions_adv=preds_2_adv, evaluate=evaluate_2,
                args=train_params, rng=rng)

    return report


def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
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
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))
    flags.DEFINE_integer('train_start', 1000, 'start of MNIST training samples')
    flags.DEFINE_integer('train_end', 1500, 'end of MNIST training samples')
    flags.DEFINE_integer('test_start', 0, 'start of MNIST test samples')
    flags.DEFINE_integer('test_end', 50, 'end of MNIST test samples')

    tf.app.run()
