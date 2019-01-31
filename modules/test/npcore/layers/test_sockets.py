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

from npcore.layer.gates import Linear, ReLU
from npcore.layer.link import Link
from npcore.layer.sockets import Dropout, BatchNorm

# ------------------------------------------------------------------------

np.random.seed(2)

# ------------------------------------------------------------------------


class TestUnitInitializers(unittest.TestCase):
    def test_init(self):
        print('Testing Socket Layers.')

        input_size = 3
        output_size = 2
        shape = (input_size, output_size)
        x_t = np.random.rand(3, 3)
        eyg_t = np.random.rand(3, 2)
        hparam = {
            'batch_size': 2,
            'eta': 1e-3,
            'l1_lambda': 1e-2,
            'l2_lambda': 1e-2,
            'momentum': 0.9,
            'beta_decay1': 0.9,
            'beta_decay2': 0.999
        }
        forward_stage = {
            'epoch': 1,
            'mode': 'learning',
            'hparam': hparam
        }
        backward_stage = {
            'epoch': 1,
            'mode': 'learning',
            'hparam': hparam
        }

        relu = ReLU(size=input_size, name='input')
        dropout = Dropout(size=input_size, name='input_dropout', pzero=0.5)
        batch_norm = BatchNorm(size=input_size, name='input_batch_norm')
        linear = Linear(size=output_size, name='output')
        link = Link(shape=shape, name='dense')

        seq = relu.connect(dropout).connect(batch_norm).connect(link).connect(linear).head

        seq.forward(forward_stage, x_t).backward(backward_stage, eyg_t)

        print(seq.tail.outputs)


if __name__ == '__main__':
    unittest.main()
