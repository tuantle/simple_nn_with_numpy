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

from npcore.optimizers import (
    Optimizer,
    SGD,
    SGDM,
    RMSprop,
    Adam
)

# ------------------------------------------------------------------------

np.random.seed(2)

# ------------------------------------------------------------------------


class TestUnitIOptimizers(unittest.TestCase):
    def test_init(self):
        print('Testing Unit Optimizers.')

        hparam = {
            'eta': 1e-3,
            'momentum': 0.9,
            'beta_decay1': 0.9,
            'beta_decay2': 0.999
        }
        e1_t = np.random.rand(2, 3)
        e2_t = np.random.rand(1, 3)

        opt = Optimizer()
        sgd = SGD()
        sgdm = SGDM()
        rmsprop = RMSprop()
        adam = Adam()

        print(opt)
        print(opt.snapshot())

        print(sgd)
        print(sgd.snapshot())
        print(sgd.compute_grad_descent_step(0, [e1_t, e2_t], hparam))

        print(sgdm)
        print(sgdm.snapshot())
        for i in range(20):
            e1_t = np.random.rand(2, 3)
            e2_t = np.random.rand(1, 3)
            sgdm.compute_grad_descent_step(i, [e1_t, e2_t], hparam)
        print(sgdm.compute_grad_descent_step(i, [e1_t, e2_t], hparam))

        print(rmsprop)
        print(rmsprop.snapshot())
        for i in range(20):
            e1_t = np.random.rand(2, 3)
            e2_t = np.random.rand(1, 3)
            rmsprop.compute_grad_descent_step(i, [e1_t, e2_t], hparam)
        print(rmsprop.compute_grad_descent_step(i, [e1_t, e2_t], hparam))

        print(adam)
        print(adam.snapshot())
        for i in range(20):
            e1_t = np.random.rand(2, 3)
            e2_t = np.random.rand(1, 3)
            adam.compute_grad_descent_step(i, [e1_t, e2_t], hparam)
        print(adam.compute_grad_descent_step(i, [e1_t, e2_t], hparam))


if __name__ == '__main__':
    unittest.main()
