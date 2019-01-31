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
import math
import numpy as np

import util.validation as validation

from model.sequencer import Sequencer
from model.nn.feed_forward import FeedForward

import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

# ------------------------------------------------------------------------

validation.DISABLE_VALIDATION = True

sns.set(context='notebook', style='white', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)

# ------------------------------------------------------------------------


class AutoencoderModel(FeedForward):
    def __init__(self, name):
        self._reports = []
        super().__init__(name=name)
        self.assign_hook(monitor=self.monitor)

    def construct(self):
        return Sequencer(name='autoencoder').add(
            'relu',
            size=8,
            name='encoder_input'
        ).add(
            'relu',
            size=6,
            name='encoder_hidden'
        ).reconfig(
            optim='sgd'
        ).add(
            'relu',
            size=4,
            name='latent'
        ).reconfig(
            optim='sgd'
        ).add(
            'relu',
            size=6,
            name='decoder_hidden'
        ).reconfig(
            optim='sgd'
        ).add(
            'linear',
            size=8,
            name='decoder_output'
        ).reconfig(
            optim='adam'
        ).reconfig_all(
            weight_init='glorot_random_normal',
            weight_reg='l1l2_elastic_net'
        )

    def monitor(self, report):
        if report['stage']['mode'] == 'learning_and_testing':
            self._reports.append(report)

    def plot(self):
        learning_losses = []
        testing_losses = []
        for report in self._reports:
            evaluation_metric = report['evaluation_metric']
            learning_losses.append(evaluation_metric['learning']['loss'])
            testing_losses.append(evaluation_metric['testing']['loss'])

        figure1 = plt.figure()
        figure1.suptitle('Evaluations')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(learning_losses, color='orangered', linewidth=1, linestyle='solid', label='Learning Loss')
        plt.plot(testing_losses, color='salmon', linewidth=1, linestyle='dotted', label='Testing Loss')
        plt.legend(fancybox=True)
        plt.grid()

        figure2 = plt.figure()
        figure2.suptitle('Training Results Per Epoch')
        plt.xlabel('Sample')

        epoch_limit = len(self._reports)
        sample_size = self._reports[0]['snapshot']['learning']['outputs'].shape[0]
        ax = plt.axes(xlim=(0, sample_size), ylim=(0, 1))
        line, = ax.plot([], [], color='blue', linewidth=1, linestyle='solid', label='Outputs')

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            y = self._reports[i]['snapshot']['learning']['outputs'][:, 0]
            x = list(range(0, y.shape[0]))
            line.set_data(x, y)
            return line,

        plt.plot(self._reports[0]['snapshot']['learning']['inputs'][:, 0], color='green', linewidth=1, linestyle='solid', label='Inputs')
        plt.plot(self._reports[0]['snapshot']['learning']['expected_outputs'][:, 0], color='red', linewidth=1, linestyle='solid', label='Expected Outputs')
        plt.legend(fancybox=True)
        plt.grid()

        anim = animation.FuncAnimation(figure2, animate, init_func=init,
                                       frames=epoch_limit, interval=48, repeat_delay=1000, blit=True)
        plt.show()

        return anim


def run_example():
    print('Simple Signal Denoising Using Autoencoder Example.')

    model = AutoencoderModel(name='Autoencoder').setup(objective='xtanh_loss')

    theta = np.linspace(-20 * np.pi, 20 * np.pi, 250)

    if not os.path.isfile('modules/examples/models/signal_denoising.json'):
        input_t = []
        expected_output_t = []
        for i in range(250):
            buff_a = []
            buff_b = []
            for j in range(1, 9):
                val = (
                    0.15 * math.sin((theta[i] / 2) + j) +
                    0.1 * math.sin((theta[i] / 3) - j) +
                    0.05 * math.sin((theta[i] / 6) + j) +
                    0.01 * math.sin((theta[i] / 9) - j) +
                    0.5)
                buff_a.append(val + np.mean(np.random.random(10)) * 0.25)
                buff_b.append(val)
            input_t.append(buff_a)
            expected_output_t.append(buff_b)
        input_t = np.array(input_t)
        expected_output_t = np.array(expected_output_t)

        print(model.summary)
        model.learn(input_t, expected_output_t, epoch_limit=100, batch_size=4, tl_split=0.2, verbose=True)
        model.save_snapshot('modules/examples/models/', save_as='signal_denoising')
        anim = model.plot()
        anim.save('modules/examples/plots/signal_denoising.gif', dpi=80, writer='pillow')
    else:
        model.load_snapshot('modules/examples/models/signal_denoising.json', overwrite=True)
        print(model.summary)

        input_t = []
        expected_output_t = []
        for i in range(250):
            buff_a = []
            buff_b = []
            for j in range(1, 9):
                val = (
                    0.15 * math.sin((theta[i] / 2) + j) +
                    0.1 * math.sin((theta[i] / 3) - j) +
                    0.05 * math.sin((theta[i] / 6) + j) +
                    0.01 * math.sin((theta[i] / 9) - j) +
                    0.5)
                buff_a.append(val + np.mean(np.random.random(10)) * 0.2)
                buff_b.append(val)
            input_t.append(buff_a)
            expected_output_t.append(buff_b)
        input_t = np.array(input_t)
        expected_output_t = np.array(expected_output_t)
        predicted_output_t = model.predict(input_t)
        error_t = (expected_output_t - predicted_output_t) / 2

        figure = plt.figure()
        figure.suptitle('Prediction Results')
        plt.xlabel('Sample')
        plt.plot(theta, input_t[:, 0], color='green', linewidth=1, linestyle='solid', label='Inputs')
        plt.plot(theta, expected_output_t[:, 0], color='red', linewidth=1, linestyle='solid', label='Expected Outputs')
        plt.plot(theta, predicted_output_t[:, 0], color='blue', linewidth=1, linestyle='solid', label='Outputs')
        plt.fill_between(theta, predicted_output_t[:, 0] + error_t[:, 0], predicted_output_t[:, 0] - error_t[:, 0], alpha=0.5, edgecolor='pink', facecolor='red', linewidth=0)
        plt.legend(fancybox=True)
        plt.grid()
        plt.show()

# ------------------------------------------------------------------------


if __name__ == '__main__':
    run_example()
