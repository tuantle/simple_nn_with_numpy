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
import unittest
import numpy as np

from npcore.layer.gates import (
    ReLU
)
from npcore.layer.sockets import (
    Dropout,
    BatchNorm
)
from model.sequencer import Sequencer
from model.nn.feed_forward import FeedForward

# ------------------------------------------------------------------------

np.random.seed(2)

# ------------------------------------------------------------------------


class FeedForwardModel(FeedForward):
    def __init__(self, name):
        super().__init__(name=name)

    def construct(self):
        return Sequencer(name='test_feed_forward').add(
            ReLU(),
            size=4,
            name='input'
        ).add(
            BatchNorm()
        ).add(
            Dropout(pzero=0.5)
        ).reconfig(
            pzero=0.25
        ).add(
            ReLU(),
            size=8,
            name='hidden1'
        ).reconfig(
            weight_init='glorot_random_normal',
            bias_init='zeros',
            optim='sgd',
        ).add(
            ReLU(),
            size=12,
            name='hidden2'
        ).reconfig(
            weight_init='glorot_random_normal',
            bias_init='zeros',
            optim='sgd',
        ).add(
            'linear',
            size=8,
            name='output'
        ).reconfig(
            weight_init='glorot_random_normal',
            bias_init='zeros',
            optim='adam'
        )

        # seq = Sequencer.create(
        #     'relu',
        #     size=4,
        #     name='input',
        # )
        # seq = Sequencer.create(
        #     Dropout(pzero=0.5)
        # )(seq)
        # # seq.reconfig(
        # #     pzero=0.5,
        # # )
        # seq = Sequencer.create(
        #     ReLU(),
        #     size=8,
        #     name='hidden1',
        # )(seq)
        # seq = Sequencer.create(
        #     'relu',
        #     size=12,
        #     name='hidden2',
        # )(seq)
        # seq = Sequencer.create(
        #     'linear',
        #     size=5,
        #     name='output',
        # )(seq)
        # seq.reconfig_all(
        #     pzero=0.5,
        #     weight_init='glorot_random_normal',
        #     bias_init='zeros',
        #     optim='adam'
        # )
        #return seq

# ------------------------------------------------------------------------


class TestUnitInitializers(unittest.TestCase):
    def test_init(self):
        print('Testing Feed Forward Model.')

    # model1 = FeedForwardModel(name='FeedForwardModel').setup(objective='mse_loss')

    # print(model1.summary)
    # print(model1.snapshot())
    # model1.save_snapshot('modules/test/model/', save_as='feed_forward')
    model2 = FeedForward().load_snapshot('modules/test/model/feed_forward.json')
    print(model2.summary)


if __name__ == '__main__':
    unittest.main()
