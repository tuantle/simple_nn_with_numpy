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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import util.validation as validation

from npcore.layer.sockets import BatchNorm
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
        self._training_snapshots = []
        super().__init__(name=name)
        self.assign_hook(monitor=self.monitor)

    def monitor(self, report):
        if report['stage']['mode'] == 'learning':
            self._reports.append(report)

    def plot(self):
        learning_losses = []
        testing_losses = []
        for report in self._reports:
            evaluation_metric = report['evaluation_metric']
            if 'learning' in evaluation_metric:
                learning_losses.append(evaluation_metric['learning']['loss'])
            if 'testing' in evaluation_metric:
                testing_losses.append(evaluation_metric['testing']['loss'])

        figure1 = plt.figure()
        figure1.suptitle('Evaluations')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if len(learning_losses) != 0:
            plt.plot(learning_losses, color='orangered', linewidth=1, linestyle='solid', label='Learning Loss')
        if len(testing_losses) != 0:
            plt.plot(testing_losses, color='salmon', linewidth=1, linestyle='dotted', label='Testing Loss')
        plt.legend(fancybox=True)
        plt.grid()

        figure2 = plt.figure()
        figure2.suptitle('Training Results Per Epoch')
        imgs = []
        for i in range(len(self._reports)):
            predicted_output_t = self._reports[i]['snapshot']['learning']['outputs'][0]
            img = (0.25 * (predicted_output_t + 1)).clip(0, 1).reshape((-1, 1, 28, 28))
            img = plt.imshow(np.squeeze(img), cmap='gray_r', origin='upper', animated=True)
            imgs.append([img])

        anim = animation.ArtistAnimation(figure2, imgs, interval=48, repeat_delay=1000, blit=True)

        plt.show()

        return anim

    def construct(self):
        img_size = 28 * 28
        return Sequencer(name='autoencoder').add(
            'relu',
            size=img_size,
        ).add(
            'relu',
            size=256,
        ).add(
            'relu',
            size=64
        ).add(
            'relu',
            size=64
        ).add(
            BatchNorm()
        ).add(
            'relu',
            size=64
        ).reconfig(
            bias_init='not_use'
        ).add(
            'relu',
            size=256
        ).add(
            'linear',
            size=img_size
        ).reconfig_all(
            weight_init='glorot_random_normal'
        )


def run_example():
    print('Simple MNIST Reconstruction Using Autoencoder Example.')

    model = AutoencoderModel(name='Autoencoder').setup(objective='algebraic_loss', optim='adam')

    if not os.path.isfile('modules/examples/models/mnist_reconstruction.json'):
        mnist_ds = DataLoader(datasets.MNIST(root='./datasets/mnist',
                                             train=False,
                                             download=True,
                                             transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                              batch_size=512,
                              shuffle=True)

        for ds in mnist_ds:
            print('Training ditgit set:')
            print(ds[1].to('cpu').numpy())
            ds_t = ds[0].to('cpu').numpy()
            mnist_img = (0.25 * (ds_t + 1)).clip(0, 1).reshape((ds_t.shape[0], 1, 28, 28))
            break

        input_t = mnist_img.reshape((mnist_img.shape[0], 28 * 28))
        expected_output_t = input_t

        print(model.summary)
        model.learn(input_t, expected_output_t, epoch_limit=100, batch_size=4, tl_split=0.0)
        model.save_snapshot('modules/examples/models/', save_as='mnist_reconstruction')
        anim = model.plot()
        anim.save('modules/examples/plots/mnist_reconstruction.gif', dpi=80, writer='pillow')
    else:
        model.load_snapshot('modules/examples/models/mnist_reconstruction.json', overwrite=True)
        print(model.summary)
        mnist_ds = DataLoader(datasets.MNIST(root='./datasets/mnist',
                                             train=False,
                                             download=True,
                                             transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                              batch_size=1,
                              shuffle=True)
        for ds in mnist_ds:
            ds0_t = ds[0].to('cpu').numpy()
            ds1_t = ds[1].to('cpu').numpy()
            print(f'Testing ditgit: {ds1_t}')
            mnist_img = (0.25 * (ds0_t + 1)).clip(0, 1).reshape((ds0_t.shape[0], 1, 28, 28))
            break

        input_t = mnist_img.reshape((mnist_img.shape[0], 28 * 28))
        expected_output_t = input_t

        predicted_output_t = model.predict(input_t)
        reconstructed__mnist_img = (0.25 * (predicted_output_t + 1)).clip(0, 1).reshape((-1, 1, 28, 28))

        figure1 = plt.figure()
        figure1.suptitle('Original MNIST')
        plt.imshow(np.squeeze(mnist_img[0]), cmap='gray_r', origin='upper')
        figure2 = plt.figure()
        figure2.suptitle('Reconstructed MNIST')
        plt.imshow(np.squeeze(reconstructed__mnist_img[0]), cmap='gray_r', origin='upper')
        plt.show()

# ------------------------------------------------------------------------


if __name__ == '__main__':
    run_example()
