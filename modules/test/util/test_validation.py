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
from util.validation import (
    MShape,
    MType,
    FType,
    OneOfType
)

# ------------------------------------------------------------------------


@FType(str, b1=int, c1=str, d1=OneOfType((int,), str))
def test_a(a1, *, b1=0, c1='c', d1=(1, 2)):
    print(a1)
    print(b1)
    print(c1)
    print(d1)


# class Test(object):
#     @property
#     def shape(self):
#         return (3, 4)
#
#     @MType(str, OneOfType(int, str))
#     def test_a(self, a, b):
#         print(a)
#         print(b)
#
#     @MType((OneOfType(str, int),), OneOfType(int, str))
#     def test_b(self, a, b):
#         print(a)
#         print(b)
#
#     @MShape(axis=-2)
#     @MType(npndarray, npndarray, c=OneOfType(npndarray, None), d=npndarray)
#     def test_c(self, a, b, *, c=None, d=npfull(shape=(3, 4), dtype=float, fill_value=0)):
#         print(a)
#         print(b)


class TestTypeValidationClass(unittest.TestCase):
    def test_init(self):
        print('Testing Type validation.')

        # Test().test_a('a', 1)
        # Test().test_b(('a', 1, 'b'), 1)
        test_a('a', b1=1, c1='c', d1=(2, 1))
        # Test().test_c(npfull(shape=(3, 4), dtype=float, fill_value=0), npfull(shape=(3, 4), dtype=float, fill_value=0))


# ------------------------------------------------------------------------


if __name__ == '__main__':
    unittest.main()
