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

from npcore.layer.layer import Layer


# ------------------------------------------------------------------------


class TestCoreLayerClass(unittest.TestCase):
    def test_init(self):
        print('Testing Core Layer.')
        shape = (1, 2)
        layer0 = Layer(shape=shape, name='l0')
        layer1 = Layer(shape=shape, name='l1n')
        layer2 = Layer(shape=shape, name='l2n')
        layer3 = Layer(shape=shape, name='l3')
        layer4 = Layer(shape=shape, name='l4')
        layer5 = Layer(shape=shape, name='l5')
        layerx = Layer(shape=shape, name='lx')
        layery = Layer(shape=shape, name='ly')
        layerz = Layer(shape=shape, name='lz')

        layer0.connect(layer1).connect(layer2).connect(layer3).connect(layer4).connect(layer5)

        layer0.from_index(2).connect(layerx, position='behind')
        layer0.from_index(5).connect(layery, position='ahead')
        layer0.from_index(4).disconnect()
        layer0.from_index(1).replace_with(layerz)

        layer = layer0.head
        print(layerz.head.name)
        while layer is not None:
            print(layer.snapshot())
            layer = layer.next

# ------------------------------------------------------------------------


if __name__ == '__main__':
    unittest.main()
