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
# distributed under the License is distributed on an "AS IS" BASIS,
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
from matplotlib import animation
from matplotlib.pyplot import cm
import seaborn as sns

# ------------------------------------------------------------------------

validation.DISABLE_VALIDATION = True

sns.set(context='notebook', style='white', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)

sample_size = 512  # number of points per class
dim = 2  # dimensionality
class_size = 2  # number of classes
training_input_t = np.zeros((sample_size * class_size, dim))  # data matrix (each row = single example)
expected_output_t = np.zeros((sample_size * class_size, class_size))  # data matrix (each row = single example)
expected_output_labels = np.zeros(sample_size * class_size, dtype='uint8')  # class labels


def generate_dataset():
    for j in range(class_size):
        ix = range(sample_size * j, sample_size * (j + 1))
        if j == 0:
            t = np.linspace(0, 0.5 * np.pi, sample_size)
            x = np.linspace(0, 1, sample_size)
            w = 0.25 * np.sin(2 * np.pi * 0.3 * t) + 0.495
            ry = np.random.uniform(w, w - 0.8, sample_size)
            training_input_t[ix] = np.c_[x, ry]
        else:
            t = np.linspace(0, 0.5 * np.pi, sample_size)
            x = np.linspace(0, 1, sample_size)
            w = 0.35 * np.sin(2 * np.pi * 0.3 * t) + 0.505
            ry = np.random.uniform(w + 0.8, w, sample_size)
            training_input_t[ix] = np.c_[x, ry]
        expected_output_t[ix] = np.array([1 if j == 0 else 0, 1 if j == 1 else 0])
        expected_output_labels[ix] = j


# ------------------------------------------------------------------------


class BinaryClassifierModel(FeedForward):
    def __init__(self, name):
        self._reports = []
        self._training_snapshots = []
        super().__init__(name=name)
        self.assign_hook(monitor=self.monitor)

    def monitor(self, report):
        if report['stage']['mode'] == 'learning_and_testing':
            self._reports.append(report)

    def on_epoch_end(self, epoch):
        (x_min, x_max) = (0, 1)
        (y_min, y_max) = (0, 1)
        (xx, yy) = np.meshgrid(np.arange(x_min, x_max, 0.005), np.arange(y_min, y_max, 0.005))
        testing_input_t = np.c_[xx.ravel(), yy.ravel()]
        predicted_output_t = self.predict(testing_input_t)
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

        figure2 = plt.figure()
        figure2.suptitle('Training Results Per Epoch')

        epoch_limit = len(self._training_snapshots)
        (x_min, x_max) = (0, 1)
        (y_min, y_max) = (0, 1)
        (xx, yy) = np.meshgrid(np.arange(x_min, x_max, 0.005), np.arange(y_min, y_max, 0.005))
        cmap_spring = cm.get_cmap('spring')
        cmap_cool = cm.get_cmap('cool')
        imgs = []
        for epoch in range(epoch_limit):
            zz = self._training_snapshots[epoch]
            im = plt.contourf(xx, yy, zz, cmap=cmap_spring)
            imgs.append(im.collections)
        plt.scatter(training_input_t[:, 0], training_input_t[:, 1], c=expected_output_labels, s=5, cmap=cmap_cool)
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
        seq = Sequencer.create(
            'swish',
            size=dim * 8,
            name='hidden2'
        )(seq)
        seq = Sequencer.create(
            'linear',
            size=class_size,
            name='output',
        )(seq)
        seq.reconfig_all(
            optim='adam',
            weight_init='glorot_random_normal'
        )
        seq.name = 'binary_classifier'
        return seq


def run_example():
    print('Simple Binary Classifier Example.')

    model = BinaryClassifierModel(name='BinaryClassifier').setup(objective='sigmoid_crossentropy_loss', metric=('loss', 'accuracy', 'f1_score'))

    if not os.path.isfile('modules/examples/models/binary_classifier.json'):
        generate_dataset()
        print(model.summary)
        model.learn(training_input_t, expected_output_t, epoch_limit=50, batch_size=32, tl_split=0.2, tl_shuffle=True)
        model.save_snapshot('modules/examples/models/', save_as='binary_classifier')
        anim = model.plot()
        anim.save('modules/examples/plots/binary_classifier.gif', dpi=80, writer='pillow')
    else:
        model.load_snapshot('modules/examples/models/binary_classifier.json', overwrite=True)
        print(model.summary)

        (x_min, x_max) = (0, 1)
        (y_min, y_max) = (0, 1)
        (xx, yy) = np.meshgrid(np.arange(x_min, x_max, 0.005), np.arange(y_min, y_max, 0.005))
        testing_input_t = np.c_[xx.ravel(), yy.ravel()]

        generate_dataset()
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
    run_example()
