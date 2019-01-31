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
    Linear,
    ReLU,
    ELU,
    SoftPlus,
    Swish,
    Sigmoid,
    Tanh,
    Algebraic
)

# ------------------------------------------------------------------------

np.random.seed(2)

# ------------------------------------------------------------------------


class TestGates(unittest.TestCase):
    def test_init(self):
        print('Testing Gate Layers.')

        z_t = np.array([[-0.2, 0.0, 0.1, 0.5, 0.9]])
        dummy_t = np.array([[0, 0, 0, 0, 0]])

        size = z_t.shape[1]
        forward_stage = {
            'epoch': 1,
            'hparam': {}
        }
        backward_stage = {
            'epoch': 1,
            'hparam': {}
        }

        print('Linear')
        linear = Linear(size=size, name='test')
        linear.forward(forward_stage, z_t)
        print(linear.outputs)
        print(linear.backward(backward_stage, dummy_t))

        print('ReLU')
        relu = ReLU(size=size, name='test')
        relu.forward(forward_stage, z_t)
        print(relu.outputs)
        print(relu.backward(backward_stage, dummy_t))

        print('ELU')
        elu = ELU(size=size, name='test')
        elu.forward(forward_stage, z_t)
        print(elu.outputs)
        print(elu.backward(backward_stage, dummy_t))

        print('SoftPlus')
        soft_plus = SoftPlus(size=size, name='test')
        print(soft_plus)
        soft_plus.forward(forward_stage, z_t)
        print(soft_plus.outputs)
        print(soft_plus.backward(backward_stage, dummy_t))

        print('Swish')
        swish = Swish(size=size, name='test')
        print(swish)
        swish.forward(forward_stage, z_t)
        print(swish.outputs)
        print(swish.backward(backward_stage, dummy_t))

        print('Sigmoid')
        sigmoid = Sigmoid(size=size, name='test')
        print(sigmoid)
        sigmoid.forward(forward_stage, z_t)
        print(sigmoid.outputs)
        print(sigmoid.backward(backward_stage, dummy_t))

        print('Tanh')
        tanh = Tanh(size=size, name='test')
        print(tanh)
        tanh.forward(forward_stage, z_t)
        print(tanh.outputs)
        print(tanh.backward(backward_stage, dummy_t))

        print('Algebraic')
        algebraic = Algebraic(size=size, name='test')
        print(algebraic)
        algebraic.forward(forward_stage, z_t)
        print(algebraic.outputs)
        print(algebraic.backward(backward_stage, dummy_t))


if __name__ == '__main__':
    unittest.main()
