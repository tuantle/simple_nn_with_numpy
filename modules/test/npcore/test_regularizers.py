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
import math
import numpy as np

from npcore.regularizers import (
    Regularizer,
    L1Lasso,
    L2Ridge,
    L1L2ElasticNet
)

# ------------------------------------------------------------------------

np.random.seed(2)

# ------------------------------------------------------------------------


class TestUnitInitializers(unittest.TestCase):
    def test_init(self):
        print('Testing Regularizers.')

        w_m = 3 * np.random.rand(3, 2)
        print(w_m)
        parameter = {
            'eta': 1e-2,
            'l1_lambda': 1e-2,
            'l2_lambda': 1e-2
        }

        reg = Regularizer()
        l1_lasso = L1Lasso()
        l2_ridge = L2Ridge()
        l1l2_elastic_net = L1L2ElasticNet()

        print(reg)
        print(reg.snapshot())
        # print(reg.compute_regularization(w_m, parameter))

        print(l1_lasso)
        print(l1_lasso.snapshot())
        print(l1_lasso.compute_regularization(0, w_m, parameter))

        print(l2_ridge)
        print(l2_ridge.snapshot())
        print(l2_ridge.compute_regularization(0, w_m, parameter))

        print(l1l2_elastic_net)
        print(l1l2_elastic_net.snapshot())
        print(l1l2_elastic_net.compute_regularization(0, w_m, parameter))


if __name__ == '__main__':
    unittest.main()
