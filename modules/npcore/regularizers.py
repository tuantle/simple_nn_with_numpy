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

import abc
import json
import numpy as np

from util.const import CONST
from util.validation import MType

# ------------------------------------------------------------------------


class REGULARIZER(CONST):
    LABEL = 'reg'
    L1_LASSO_LABEL = 'l1_lasso'
    L2_RIDGE_LABEL = 'l2_ridge'
    L1L2_ELASTIC_NET_LABEL = 'l1l2_elastic_net'

    DEFAULT_L1_LAMBDA = 1e-3
    DEFAULT_L2_LAMBDA = 1e-3


# ------------------------------------------------------------------------


class Regularizer(type):
    """
    A metaclass for a regularizer class
    """
    def __getattr__(cls, key):
        return cls[key]

    def __setattr__(cls, key, value):
        raise ValueError('Cannot set value to an regularizer class.')

    @property
    def label(cls):
        """
        Get regularizer label.
        Returns:
            str
        """
        return cls._label


# ------------------------------------------------------------------------


class Regularizer(object, metaclass=Regularizer):
    _label = REGULARIZER.LABEL
    """
    A base regularizer class.
    """

    def __str__(self):
        return self.label

    # ------------------------------------------------------------------------

    @property
    def label(self):
        """
        Get regularizer label.
        Returns:
            str
        """
        return type(self).label

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return regularizer state snapshot as a dict.
        Arguments:
            as_json: set to True to convert and return dict as JSON
            beautify_json: set to True to beautify JSON
        Returns:
            dict
        """
        snapshot = {
            'label': self.label,
            'base_label': Regularizer.label,
        }
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()

    @abc.abstractmethod
    def compute_regularization(self):
        """
        Compute the regularized weight matrix. Not implemented.
        """
        pass


# ------------------------------------------------------------------------


class L1Lasso(Regularizer):
    _label = REGULARIZER.L1_LASSO_LABEL

    # ------------------------------------------------------------------------

    @MType(int, np.ndarray, dict)
    def compute_regularization(self, epoch, w_m, hparam):
        """
        Compute the weight matrix regularization using lasso or l1 method.
        Arguments:
            epoch:
            w_m: weight matrix
            hparam: hold eta or learning rate and lambda or regularization rate
        Returns:
            matrix
        """
        eta = hparam['eta']
        l1_lambda = hparam.get('l1_lambda', REGULARIZER.DEFAULT_L1_LAMBDA)
        return eta * l1_lambda * np.vectorize(lambda element: (element > 0 and 1) or (element < 0 and -1))(w_m)


# ------------------------------------------------------------------------


class L2Ridge(Regularizer):
    _label = REGULARIZER.L2_RIDGE_LABEL

    # ------------------------------------------------------------------------

    @MType(int, np.ndarray, dict)
    def compute_regularization(self, epoch, w_m, hparam):
        """
        Compute the weight matrix regularization using ridge or l2 method.
        Arguments:
            epoch:
            w_m: weight matrix
            hparam: hold eta or learning rate and lambda or regularization rate
        Returns:
            matrix
        """
        eta = hparam['eta']
        l2_lambda = hparam.get('l2_lambda', REGULARIZER.DEFAULT_L2_LAMBDA)
        return eta * l2_lambda * w_m


# ------------------------------------------------------------------------


class L1L2ElasticNet(Regularizer):
    _label = REGULARIZER.L1L2_ELASTIC_NET_LABEL

    # ------------------------------------------------------------------------

    @MType(int, np.ndarray, dict)
    def compute_regularization(self, epoch, w_m, hparam):
        """
        Compute the weight matrix regularization using elastic net or l1l2 method.
        Arguments:
            epoch:
            w_m: weight matrix
            hparam: hold eta or learning rate and lambda or regularization rate
        Returns:
            matrix
        """
        eta = hparam['eta']
        l1_lambda = hparam.get('l1_lambda', REGULARIZER.DEFAULT_L1_LAMBDA)
        l2_lambda = hparam.get('l2_lambda', REGULARIZER.DEFAULT_L2_LAMBDA)
        return eta * (l2_lambda * w_m + l1_lambda * np.vectorize(lambda element: (element > 0 and 1) or (element < 0 and -1))(w_m))
