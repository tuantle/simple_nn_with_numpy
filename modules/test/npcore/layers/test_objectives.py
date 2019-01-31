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
from npcore.layer.objectives import (
    Objective,
    MSELoss,
    MAELoss,
    LogCoshLoss,
    XTanhLoss,
    AlgebraicLoss,
    SigmoidCrossentropyLoss,
    SoftmaxCrossentropyLoss
)

# ------------------------------------------------------------------------

np.random.seed(2)

# ------------------------------------------------------------------------


class TestUnitInitializers(unittest.TestCase):
    def test_init(self):
        print('Testing Regression Objective Layers.')

        input_size = 3
        output_size = 2
        shape = (input_size, output_size)
        x_t = np.random.rand(3, 3)
        y_t = np.random.rand(3, 2)
        stage = {
            'epoch': 1,
            'mode': 'learning',
            'hparam': {
                'batch_size': 2,
                'eta': 1e-3,
                'l1_lambda': 1e-2,
                'l2_lambda': 1e-2,
                'momentum': 0.9,
                'beta_decay1': 0.9,
                'beta_decay2': 0.999
            }
        }

        relu = ReLU(size=input_size, name='input')
        linear = Linear(size=output_size, name='output')
        link = Link(shape=shape, name='dense')

        mae = MAELoss(size=output_size, name='objective')

        seq = relu.connect(link).connect(linear).connect(mae).head

        seq.forward(stage, x_t).evaluate(y_t).backward(stage)

        print(seq.tail.outputs)
        print(seq.tail.evaluation_metric)

        print('Testing Category Classification Objective Layers.')

        input_size = 3
        output_size = 2
        shape = (input_size, output_size)
        x_t = np.random.rand(2, 3)
        y_t = np.array([[0, 1], [1, 0]])
        stage = {
            'epoch': 1,
            'mode': 'learning',
            'hparam': {
                'batch_size': 2,
                'eta': 1e-3,
                'l1_lambda': 1e-2,
                'l2_lambda': 1e-2,
                'momentum': 0.9,
                'beta_decay1': 0.9,
                'beta_decay2': 0.999
            }
        }

        relu = ReLU(size=input_size, name='input')
        linear = Linear(size=output_size, name='output')
        link = Link(shape=shape, name='dense')

        cce = SoftmaxCrossentropyLoss(size=output_size, name='objective')

        seq = relu.connect(link).connect(linear).connect(cce).head

        seq.forward(stage, x_t).evaluate(y_t).backward(stage)

        print(seq.tail.outputs)
        print(seq.tail.evaluation_metric)


if __name__ == '__main__':
    unittest.main()
