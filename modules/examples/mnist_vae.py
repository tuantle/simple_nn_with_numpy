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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from npcore.layer.sockets import Socket
from npcore.layer.objectives import Objective
from npcore.initializers import (
    Zeros,
    RandomNormal,
    GlorotRandomNormal
)
from npcore.optimizers import SGD
from model.sequencer import Sequencer
from model.nn.feed_forward import FeedForward

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import animation
import seaborn as sns

# ------------------------------------------------------------------------

sns.set(context='notebook', style='white', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)

# ------------------------------------------------------------------------


class Sampler(Socket):
    _label = 'reparameterizer'

    def __init__(self, *, shape=):
        super().__init__(size=size, name='reparameterizer')
        self._eps_m = RandomNormal(mean=0.0, variance=0.5)((1, 4))
        self._mu_w_m = GlorotRandomNormal()(size, 4)
        self._mu_b_v = Zeros()((1, 4))
        self._sigma_w_m = GlorotRandomNormal()(size, 4)
        self._sigma_b_b = Zeros()((1, 4))
        self._optim = SGD()

    def compute_forward_ops(self, stage, a_t, *, residue={}):
        # slice_index = int(self.size / 2)
        #
        # (mu_t, log_sigma_t) = (a_t[:, slice_index:], a_t[:, :slice_index])
        (mu_t, log_sigma_t) = (a_t.copy(), a_t.copy())
        residue['reparameterizer'] = (mu_t, log_sigma_t)

        a_t = mu_t + np.exp(log_sigma_t / 2) * self._eps_m
        return (a_t, residue)

    def compute_backward_ops(self, stage, eag_t, *, residue={}):
        # slice_index = int(self.size / 2)
        #
        # (mu_eg_t, log_sigma_eg_t) = residue['reparameterizer']
        # eag_t[:, slice_index:] = mu_eg_t * eag_t[:, slice_index:]
        # eag_t[:, :slice_index] = log_sigma_eg_t * eag_t[:, :slice_index]

        (mu_eg_t, log_sigma_eg_t) = residue['reparameterizer']
        eag_t *= mu_eg_t
        eag_t *= log_sigma_eg_t
        return (eag_t, residue)


class VAELoss(Objective):
    _label = 'vae_loss'

    # ------------------------------------------------------------------------

    def compute_loss(self, y_t, y_prime_t, *, residue={}):
        ey_t = y_t - y_prime_t

        (mu_t, log_sigma_t) = residue['reparameterizer']
        ey_t = y_t - y_prime_t
        mse = np.square(ey_t)
        kld = np.exp(log_sigma_t) + np.square(mu_t) - 1 - log_sigma_t
        ly_t = mse

        self._evaluation['metric']['loss'] += mse.mean() + kld.mean()

        return (ly_t, residue)

    def compute_loss_grad(self, y_t, y_prime_t, *, residue={}):
        ey_t = y_t - y_prime_t
        eyg_t = 2 * ey_t

        (mu_t, log_sigma_t) = residue['reparameterizer']

        mu_eg_t = np.ones_like(mu_t)
        log_sigma_eg_t = np.exp(log_sigma_t / 2) / 2
        residue['reparameterizer'] = (mu_eg_t, log_sigma_eg_t)
        return (eyg_t, residue)

    def compute_evaluation_metric(self, y_t, y_prime_t, ly_t, evaluation_metric):
        # if 'loss' in evaluation_metric:
        #     evaluation_metric['loss'] += 0.5 * ly_t.mean()

        return evaluation_metric


class VariationalAutoencoderModel(FeedForward):
    def __init__(self, name):
        super().__init__(name=name)
        self._reports = []

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
        return Sequencer(name='encoder').add(
            'relu',
            size=img_size
        ).add(
            'relu',
            size=256
        ).add(
            'linear',
            size=64
        ).add(
            Sampler(),
            shape=(64, 32)
        ).add(
            'relu',
            size=32
        ).add(
            'relu',
            size=64
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
    print('Simple MNIST Generative Using Variational Autoencoder Example.')

    model = VariationalAutoencoderModel(name='VariationalAutoencoder').setup(objective=VAELoss(), optim='adam')

    if not os.path.isfile('modules/examples/models/mnist_vae.json'):
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
        model.learn(input_t, expected_output_t, epoch_limit=15, batch_size=4, tl_split=0.0)
        model.save_snapshot('modules/examples/models/', save_as='mnist_vae')
        anim = model.plot()
        anim.save('modules/examples/plots/mnist_vae.gif', dpi=80, writer='pillow')
    else:
        model.load_snapshot('modules/examples/models/mnist_vae.json', overwrite=True)
        print(model.summary)

    mnist_ds = DataLoader(datasets.MNIST(root='./datasets/mnist',
                                         train=False,
                                         download=True,
                                         transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                          batch_size=1,
                          shuffle=True)
    for ds in mnist_ds:
        ds0_t = ds[0].to('cpu').numpy()
        ds1_t = ds[0].to('cpu').numpy()
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
