#!/usr/bin/env python
#
# Copyright 2016-present Tuan Le.
#
# Licensed under the MIT License.
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://opensource.org/licenses/mit-license.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------
#
# Author Tuan Le (tuan.t.lei@gmail.com)
#
# ------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import env
import os
import numpy as np

import util.validation as validation

from model.sequencer import Sequencer
from model.nn.feed_forward import FeedForward

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import animation
import seaborn as sns
# ------------------------------------------------------------------------

validation.DISABLE_VALIDATION = True

sns.set(context='notebook', style='white', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)

sample_size = 512  # number of points per class
dim = 2  # dimensionality
class_size = 3  # number of classes
training_input_t = np.zeros((sample_size * class_size, dim))  # data matrix (each row = single example)
expected_output_t = np.zeros((sample_size * class_size, class_size))  # data matrix (each row = single example)
expected_output_labels = np.zeros(sample_size * class_size, dtype='uint8')  # class labels


def generate_sprirals():
    for j in range(class_size):
        ix = range(sample_size * j, sample_size * (j + 1))
        r = np.linspace(0.01, 0.4, sample_size) + np.random.randn(sample_size) * 0.01  # radius
        t = np.linspace(j * np.pi, (j + 1) * np.pi, sample_size) + np.random.randn(sample_size) * 0.2  # theta
        training_input_t[ix] = np.c_[r * np.sin(2 * np.pi * 0.1125 * t) + 0.5, r * np.cos(2 * np.pi * 0.1125 * t) + 0.5]
        expected_output_t[ix] = np.array([1 if j == 0 else 0, 1 if j == 1 else 0, 1 if j == 2 else 0])
        expected_output_labels[ix] = j


def generate_clumps():
    for j in range(class_size):
        ix = range(sample_size * j, sample_size * (j + 1))
        rx = 0.5 + np.random.randn(sample_size) * 0.05
        ry = 0.5 + np.random.randn(sample_size) * 0.05
        training_input_t[ix] = np.c_[0.2 * np.sin(2 * np.pi * 0.65 * (j + 1)) + rx, 0.2 * np.cos(2 * np.pi * 0.35 * (j + 1)) + ry]
        expected_output_t[ix] = np.array([1 if j == 0 else 0, 1 if j == 1 else 0, 1 if j == 2 else 0])
        expected_output_labels[ix] = j


def generate_rings():
    for j in range(class_size):
        ix = range(sample_size * j, sample_size * (j + 1))
        r = 0.125 * (j + 1) + np.random.randn(sample_size) * 0.025
        t = np.linspace(j * 4, (j + 1) * 4, sample_size) + np.random.randn(sample_size) * 0.4  # theta
        training_input_t[ix] = np.c_[r * np.sin(2 * np.pi * t) + 0.5, r * np.cos(2 * np.pi * t) + 0.5]
        expected_output_t[ix] = np.array([1 if j == 0 else 0, 1 if j == 1 else 0, 1 if j == 2 else 0])
        expected_output_labels[ix] = j


def generate_crater():
    for j in range(class_size):
        ix = range(sample_size * j, sample_size * (j + 1))
        r = 0.15 * j + np.random.randn(sample_size) * 0.04
        t = np.linspace(j * 4, (j + 1) * 4, sample_size) + np.random.randn(sample_size) * 0.1  # theta
        training_input_t[ix] = np.c_[r * np.sin(2 * np.pi * t) + 0.5, r * np.cos(2 * np.pi * t) + 0.5]
        expected_output_t[ix] = np.array([1 if j == 0 else 0, 1 if j == 1 else 0, 1 if j == 2 else 0])
        expected_output_labels[ix] = j


def generate_pie():
    for j in range(class_size):
        ix = range(sample_size * j, sample_size * (j + 1))
        r = np.random.uniform(low=0, high=0.25, size=sample_size)
        t = np.linspace(j, j + 1, sample_size)
        training_input_t[ix] = np.c_[r * np.sin(2 * np.pi * t / 3) + 0.5, r * np.cos(2 * np.pi * t / 3) + 0.5]
        expected_output_t[ix] = np.array([1 if j == 0 else 0, 1 if j == 1 else 0, 1 if j == 2 else 0])
        expected_output_labels[ix] = j

# ------------------------------------------------------------------------


class CategoryClassifierModel(FeedForward):
    def __init__(self, name):
        self._reports = []  # monitoring data for every epoch during training phase
        self._training_snapshots = []  # output result data for every epoch during training phase
        super().__init__(name=name)
        self.assign_hook(monitor=self.monitor)  # attach monitoring hook

    def monitor(self, report):
        #  this monitor hook is called at the end of every epoch during training
        if report['stage']['mode'] == 'learning_and_testing':
            self._reports.append(report)  # save the training report of current epoch

    def on_epoch_end(self, epoch):
        #  save the prediction output at the end of every training epoch for making cool plots later on
        (x_min, x_max) = (0, 1)
        (y_min, y_max) = (0, 1)
        (xx, yy) = np.meshgrid(np.arange(x_min, x_max, 0.005), np.arange(y_min, y_max, 0.005))
        testing_input_t = np.c_[xx.ravel(), yy.ravel()]
        predicted_output_t = self.predict(testing_input_t)  # get the current prediction for this epoch
        zz = predicted_output_t.argmax(axis=1).reshape(xx.shape)
        self._training_snapshots.append(zz)

    def plot(self):
        learning_losses = []
        testing_losses = []
        learning_accuracies = []
        testing_accuracies = []
        for report in self._reports:
            evaluation_metric = report['evaluation_metric']
            learning_losses.append(evaluation_metric['learning']['loss'])
            testing_losses.append(evaluation_metric['testing']['loss'])
            learning_accuracies.append(evaluation_metric['learning']['accuracy'])
            testing_accuracies.append(evaluation_metric['testing']['accuracy'])

        # plotting training loss and accuracy
        figure1 = plt.figure()
        figure1.suptitle('Evaluations')
        plt.subplot(2, 1, 1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(learning_losses, color='orangered', linewidth=1, linestyle='solid', label='Learning Loss')
        plt.plot(testing_losses, color='salmon', linewidth=1, linestyle='dotted', label='Testing Loss')
        plt.legend(fancybox=True)
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(learning_accuracies, color='deepskyblue', linewidth=1, linestyle='solid', label='Learning Accuracy')
        plt.plot(testing_accuracies, color='aqua', linewidth=1, linestyle='dotted', label='Testing Accuracy')
        plt.legend(fancybox=True)
        plt.grid()

        # plotting prediction result per training epoch
        figure2 = plt.figure()
        figure2.suptitle('Training Results Per Epoch')
        epoch_limit = len(self._training_snapshots)
        (x_min, x_max) = (0, 1)
        (y_min, y_max) = (0, 1)
        (xx, yy) = np.meshgrid(np.arange(x_min, x_max, 0.005), np.arange(y_min, y_max, 0.005))
        imgs = []
        for epoch in range(epoch_limit):
            zz = self._training_snapshots[epoch]
            im = plt.contourf(xx, yy, zz, cmap=cm.get_cmap('spring'))
            imgs.append(im.collections)
        plt.scatter(training_input_t[:, 0], training_input_t[:, 1], c=expected_output_labels, s=5, cmap=cm.get_cmap('cool'))
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.grid()

        anim = animation.ArtistAnimation(figure2, imgs, interval=48, repeat_delay=1000, repeat=True)

        plt.show()

        return anim

    def construct(self):
        seq = Sequencer.create(
            'swish',
            size=dim,
            name='input'
        )
        seq = Sequencer.create(
            'swish',
            size=dim * 16,
            name='hidden1'
        )(seq)
        seq.reconfig(
            optim='sgd'
        )
        seq = Sequencer.create(
            'swish',
            size=dim * 12,
            name='hidden2'
        )(seq)
        seq.reconfig(
            optim='adam'
        )
        seq = Sequencer.create(
            'linear',
            size=class_size,
            name='output',
        )(seq)
        seq.reconfig(
            optim='sgd'
        ).reconfig_all(
            weight_init='glorot_random_normal',
            weight_reg='l1l2_elastic_net',
            bias_init='zeros'
        )
        seq.name = 'category_classifier'
        return seq


def run_rings_example():
    print('Simple Category Classifier Rings Example.')
    model = CategoryClassifierModel(name='CategoryClassifier').setup(objective='softmax_crossentropy_loss',
                                                                     metric=('loss', 'accuracy'))
    if not os.path.isfile('modules/examples/models/category_classifier_rings.json'):
        generate_rings()
        print(model.summary)
        model.learn(training_input_t, expected_output_t, epoch_limit=50, batch_size=32, tl_split=0.2, tl_shuffle=True)
        model.save_snapshot('modules/examples/models/', save_as='category_classifier_rings')
        anim = model.plot()
        anim.save('modules/examples/plots/category_classifier_rings.gif', dpi=80, writer='pillow')
    else:
        model.load_snapshot('modules/examples/models/category_classifier_rings.json', overwrite=True)
        print(model.summary)

    (x_min, x_max) = (0, 1)
    (y_min, y_max) = (0, 1)
    (xx, yy) = np.meshgrid(np.arange(x_min, x_max, 0.005), np.arange(y_min, y_max, 0.005))
    testing_input_t = np.c_[xx.ravel(), yy.ravel()]

    generate_rings()
    predicted_output_t = model.predict(testing_input_t)
    zz = predicted_output_t.argmax(axis=1).reshape(xx.shape)

    figure = plt.figure()
    figure.suptitle('Classification Prediction Results')

    plt.contourf(xx, yy, zz, cmap=cm.get_cmap('spring'), alpha=0.8)
    plt.scatter(training_input_t[:, 0], training_input_t[:, 1], c=expected_output_labels, s=5, cmap=cm.get_cmap('cool'))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.grid()
    plt.show()


def run_spirals_example():
    print('Simple Category Classifier Spirals Example.')
    model = CategoryClassifierModel(name='CategoryClassifier').setup(objective='softmax_crossentropy_loss',
                                                                     metric=('loss', 'accuracy'))

    if not os.path.isfile('modules/examples/models/category_classifier_spirals.json'):
        generate_sprirals()
        print(model.summary)
        model.learn(training_input_t, expected_output_t, epoch_limit=50, batch_size=32, tl_split=0.2, tl_shuffle=True)
        model.save_snapshot('modules/examples/models/', save_as='category_classifier_spirals')
        anim = model.plot()
        anim.save('modules/examples/plots/category_classifier_spirals.gif', dpi=80, writer='pillow')
    else:
        model.load_snapshot('modules/examples/models/category_classifier_spirals.json', overwrite=True)
        print(model.summary)

    (x_min, x_max) = (0, 1)
    (y_min, y_max) = (0, 1)
    (xx, yy) = np.meshgrid(np.arange(x_min, x_max, 0.005), np.arange(y_min, y_max, 0.005))
    testing_input_t = np.c_[xx.ravel(), yy.ravel()]

    generate_sprirals()
    predicted_output_t = model.predict(testing_input_t)
    zz = predicted_output_t.argmax(axis=1).reshape(xx.shape)

    figure = plt.figure()
    figure.suptitle('Classification Prediction Results')

    cmap_spring = cm.get_cmap('spring')
    cmap_cool = cm.get_cmap('cool')

    plt.contourf(xx, yy, zz, cmap=cmap_spring, alpha=0.8)
    plt.scatter(training_input_t[:, 0], training_input_t[:, 1], c=expected_output_labels, s=5, cmap=cmap_cool)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.grid()
    plt.show()


def run_pie_example():
    print('Simple Category Classifier Pie Example.')
    model = CategoryClassifierModel(name='CategoryClassifier').setup(objective='softmax_crossentropy_loss',
                                                                     metric=('loss', 'accuracy'))

    if not os.path.isfile('modules/examples/models/category_classifier_pie.json'):
        generate_pie()
        print(model.summary)
        model.learn(training_input_t, expected_output_t, epoch_limit=50, batch_size=32, tl_split=0.2, tl_shuffle=True)
        model.save_snapshot('modules/examples/models/', save_as='category_classifier_pie')
        anim = model.plot()
        anim.save('modules/examples/plots/category_classifier_pie.gif', dpi=80, writer='pillow')
    else:
        model.load_snapshot('modules/examples/models/category_classifier_pie.json', overwrite=True)
        print(model.summary)

        (x_min, x_max) = (0, 1)
        (y_min, y_max) = (0, 1)
        (xx, yy) = np.meshgrid(np.arange(x_min, x_max, 0.005), np.arange(y_min, y_max, 0.005))
        testing_input_t = np.c_[xx.ravel(), yy.ravel()]

        generate_pie()
        predicted_output_t = model.predict(testing_input_t)
        zz = predicted_output_t.argmax(axis=1).reshape(xx.shape)

        figure = plt.figure()
        figure.suptitle('Classification Prediction Results')

        cmap_spring = cm.get_cmap('spring')
        cmap_cool = cm.get_cmap('cool')

        plt.contourf(xx, yy, zz, cmap=cmap_spring, alpha=0.8)
        plt.scatter(training_input_t[:, 0], training_input_t[:, 1], c=expected_output_labels, s=5, cmap=cmap_cool)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.grid()
        plt.show()

# ------------------------------------------------------------------------


if __name__ == '__main__':
    run_rings_example()
    # run_spirals_example()
    # run_pie_example()
