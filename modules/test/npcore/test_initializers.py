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

from npcore.initializers import (
    Initializer,
    Zeros, Ones, Constant, Identity,
    RandomBinary, RandomOrthonormal,
    RandomUniform, RandomNormal,
    GlorotRandomUniform, GlorotRandomNormal
)

# ------------------------------------------------------------------------

np.random.seed(2)

# ------------------------------------------------------------------------


class TestUnitInitializers(unittest.TestCase):
    def test_init(self):
        print('Testing Unit Initializers.')

        shape = (3, 3)
        seed = 2

        init = Initializer()
        zeros = Zeros()
        ones = Ones()
        const = Constant(0.5)
        identity = Identity()
        rbinary = RandomBinary(seed=seed)
        rortho = RandomOrthonormal(seed=seed)
        runiform = RandomUniform(seed=seed, min=0.45, max=0.55)
        rnormal = RandomNormal(seed=seed, mean=0.5, variance=0.1)
        gruniform = GlorotRandomUniform(seed=seed)
        grnormal = GlorotRandomNormal(seed=seed)

        print(init)

        print(zeros)
        print(zeros(shape))
        print(zeros.snapshot())

        print(ones)
        print(ones(shape))
        print(ones.snapshot())

        print(const)
        print(const(shape))
        print(const.snapshot())

        print(identity)
        print(identity(shape))
        print(identity.snapshot())

        print(rbinary)
        print(rbinary(shape, pzero=0.5))
        print(rbinary.snapshot())

        print(rortho)
        print(rortho(shape))
        print(rortho.snapshot())

        print(runiform)
        print(runiform(shape))
        print(runiform.snapshot())

        print(rnormal)
        print(rnormal(shape))
        print(rnormal.snapshot())

        print(gruniform)
        print(gruniform(shape))
        print(gruniform.snapshot())

        print(grnormal)
        print(grnormal(shape))
        print(grnormal.snapshot())


if __name__ == '__main__':
    unittest.main()
