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
from __future__ import print_function

# ------------------------------------------------------------------------


class CONST(type):
    """
    A metaclass for CONST class.
    """
    def __getattr__(cls, key):
        return cls[key]

    def __setattr__(cls, key, value):
        raise ValueError('Cannot set value to a class constant.')


class CONST(object, metaclass=CONST):
    """
    A const class where calling __setattr__ will raise an exception error.
    """

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        raise TypeError('Cannot set value to a constant.')
