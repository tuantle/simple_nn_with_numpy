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
import math
import numpy as np

from util.const import CONST
from util.validation import MType

# ------------------------------------------------------------------------


class OPTIMIZER(CONST):
    LABEL = 'optim'
    SGD_LABEL = 'sgd'
    SGDM_LABEL = 'sgdm'
    RMSPROP_LABEL = 'rmsprop'
    ADAM_LABEL = 'adam'

    DEFAULT_ETA = 1e-3
    DEFAULT_ETA_DECAY = 0.9
    DEFAULT_BETA_DECAY1 = 0.9
    DEFAULT_BETA_DECAY2 = 0.999
    DEFAULT_MOMENTUM = 0.9


# ------------------------------------------------------------------------


class Optimizer(type):
    """
    A metaclass for an optimizer class
    """
    def __getattr__(cls, key):
        return cls[key]

    def __setattr__(cls, key, value):
        raise ValueError('Cannot set value to an optimizer class.')

    @property
    def label(cls):
        """
        Get optimizer label.
        Returns:
            str
        """
        return cls._label


# ------------------------------------------------------------------------


class Optimizer(object, metaclass=Optimizer):
    _label = OPTIMIZER.LABEL
    """
    Abtraction of a base optimizer. Manages the stochastic gradient descent optimizations.
    """

    def __str__(self):
        return self.label

    # ------------------------------------------------------------------------

    @property
    def label(self):
        """
        Get optimizer label.
        Returns:
            str
        """
        return type(self).label

    @abc.abstractmethod
    def reset(self):
        """
        Reset internal states.
        """
        pass

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return optimizer as a snapshot dict data
        Arguments:
            as_json -
            beautify_json -
        Returns:
            snapshot
        """
        snapshot = {
            'label': self.label,
            'base_label': Optimizer.label
        }
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()

    @abc.abstractmethod
    def compute_grad_descent_step(self):
        """
        Compute gradient descent step update optimization. Not implemented
        """
        pass

# ------------------------------------------------------------------------


class SGD(Optimizer):
    _label = OPTIMIZER.SGD_LABEL
    """
    Optimization using stochastic gradient descent update rule.
    """

    # ------------------------------------------------------------------------

    def reset(self):
        """
        Reset internal states.
        """
        pass

    @MType(int, [np.ndarray], dict)
    def compute_grad_descent_step(self, epoch, egs, hparam):
        """
        Implement stochastic gradient descent update formula. Compute gradient step delta tensor.
        Arguments:
            epoch:
            egs: a list of gradient error tensors
            hparam: hold eta value
        Returns:
            list
        """
        eta = hparam['eta']
        return [eta * eg_t for eg_t in egs]


# ------------------------------------------------------------------------


class SGDM(Optimizer):
    _label = OPTIMIZER.SGDM_LABEL
    """
    Optimization using stochastic gradient descent with momentum update rule.
    """
    def __init__(self):
        self._velocities = []
        super().__init__()

    # ------------------------------------------------------------------------

    def reset(self):
        """
        Reset internal states.
        """
        super().reset()
        self._velocities = [np.zeros_like(velocity_t, dtype=velocity_t.dtype) for velocity_t in self._velocities]

    @MType(int, [np.ndarray], dict)
    def compute_grad_descent_step(self, epoch, egs, hparam):
        """
        Implement stochastic gradient descent with momentum update formula. Compute gradient step delta tensor.
        Arguments:
            epoch:
            egs: a list of gradient error tensors
            hparam: hold eta value
        Returns:
            list
        """
        eta = hparam['eta']
        momentum = hparam.get('momentum', OPTIMIZER.DEFAULT_MOMENTUM)

        if len(self._velocities) != len(egs):
            self._momentums = [momentum for eg_t in egs]
            self._velocities = [np.zeros_like(eg_t, dtype=eg_t.dtype) for eg_t in egs]

        self._velocities = [momentum * velocity_t + (1 - momentum) * eg_t for (velocity_t, eg_t) in zip(self._velocities, egs)]

        return [eta * velocity_t for velocity_t in self._velocities]


# ------------------------------------------------------------------------


class RMSprop(Optimizer):
    _label = OPTIMIZER.RMSPROP_LABEL
    """
    RMSprop update rule, which uses a moving average of squared gradient tensors to set adaptive per-parameter eta value.
    """
    def __init__(self):
        self._moving_means = []
        super().__init__()

    # ------------------------------------------------------------------------

    def reset(self):
        """
        Reset internal states.
        """
        super().reset()
        self._moving_means = [np.zeros_like(moving_mean_t, dtype=moving_mean_t.dtype) for moving_mean_t in self._moving_means]

    @MType(int, [np.ndarray], dict)
    def compute_grad_descent_step(self, epoch, egs, hparam):
        """
        Implement rmsprop update formula. Compute gradient step delta tensor.
        Arguments:
            epoch:
            egs: a list of gradient error tensors
            hparam: hold eta value
        Returns:
            list
        """
        eta = hparam['eta']
        beta_decay = hparam.get('beta_decay1', OPTIMIZER.DEFAULT_BETA_DECAY1)

        if len(self._moving_means) != len(egs):
            self._moving_means = [np.zeros_like(eg_t, dtype=eg_t.dtype) for eg_t in egs]

        self._moving_means = [beta_decay * moving_mean_t + (1 - beta_decay) * np.square(eg_t)
                              for (moving_mean_t, eg_t) in zip(self._moving_means, egs)]

        return [eta * eg_t / (np.sqrt(moving_mean_t) + 1e-12)
                for (moving_mean_t, eg_t) in zip(self._moving_means, egs)]


# ------------------------------------------------------------------------


class Adam(Optimizer):
    _label = OPTIMIZER.ADAM_LABEL
    """
    Optimization using Adam update rule, which incorporates moving averages of both the gradient and its square and a bias correction term.
    """
    def __init__(self):
        self._moving_means = []
        self._moving_sqr_means = []
        super().__init__()

    # ------------------------------------------------------------------------

    def reset(self):
        """
        Reset internal states.
        """
        super().reset()
        self._moving_means = [np.zeros_like(moving_mean_t, dtype=moving_mean_t.dtype)
                              for moving_mean_t in self._moving_means]
        self._moving_sqr_means = [np.zeros_like(moving_sqr_mean_t, dtype=moving_sqr_mean_t.dtype)
                                  for moving_sqr_mean_t in self._moving_sqr_means]

    @MType(int, [np.ndarray], dict)
    def compute_grad_descent_step(self, epoch, egs, hparam):
        """
        Implement adam update formula. Compute gradient step delta tensor.
        Arguments:
            epoch:
            egs: a list of gradient error tensors
            hparam: hold eta value
        Returns:
            list
        """
        eta = hparam['eta']
        beta_decay1 = hparam.get('beta_decay1', OPTIMIZER.DEFAULT_BETA_DECAY1)
        beta_decay2 = hparam.get('beta_decay2', OPTIMIZER.DEFAULT_BETA_DECAY2)

        if len(self._moving_means) != len(egs) or len(self._moving_sqr_means) != len(egs):
            self._moving_means = [np.zeros_like(eg_t, dtype=eg_t.dtype) for eg_t in egs]
            self._moving_sqr_means = [np.zeros_like(eg_t, dtype=eg_t.dtype) for eg_t in egs]

        bias_correction1 = 1 / (1 - math.pow(beta_decay1, epoch + 1))
        bias_correction2 = 1 / (1 - math.pow(beta_decay2, epoch + 1))

        self._moving_means = [beta_decay1 * moving_mean_t + (1 - beta_decay1) * eg_t
                              for (moving_mean_t, eg_t) in zip(self._moving_means, egs)]
        self._moving_sqr_means = [beta_decay2 * moving_sqr_mean_t + (1 - beta_decay2) * np.square(eg_t)
                                  for (moving_sqr_mean_t, eg_t) in zip(self._moving_sqr_means, egs)]

        return [eta * (moving_mean_t * bias_correction1) / (np.sqrt(moving_sqr_mean_t * bias_correction2) + 1e-12)
                for (moving_mean_t, moving_sqr_mean_t) in zip(self._moving_means, self._moving_sqr_means)]
