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

import abc
import json
import warnings
import numpy as np

from util.const import CONST
from util.validation import (
    MShape,
    MType,
    OneOfType
)
from npcore.layer.layer import Layer
from npcore.initializers import Constant

# ------------------------------------------------------------------------


class GATE(CONST):
    LABEL = 'gate'
    LINEAR_LABEL = 'linear'
    NONLINEAR_LABEL = 'nonlinear'
    RELU_LABEL = 'relu'
    LEAKY_RELU_LABEL = 'leaky_relu'
    ELU_LABEL = 'elu'
    SWISH_LABEL = 'swish'
    SOFT_PLUS_LABEL = 'soft_plus'
    SIGMOID_LABEL = 'sigmoid'
    TANH_LABEL = 'tanh'
    ALGEBRAIC_LABEL = 'algebraic'
    ARRANGEMENT = ('0', '1:2')

    LEAKY_RELU_ALPHA = 1e-2
    ELU_ALPHA = 1.67326

    SOFT_PLUS_ALPHA = 10
    SOFT_PLUS_GAMMA = 0.1
    SOFT_PLUS_BETA = 1e-3

    SWISH_ALPHA = 1
    SWISH_GAMMA = 3e-3
    SWISH_BETA = 3e-3
    # SWISH_ALPHA = 6
    # SWISH_GAMMA = 3e-2
    # SWISH_BETA = 3e-4

    SIGMOID_ALPHA = 1
    SIGMOID_GAMMA = 1
    TANH_ALPHA = 1
    # TANH_GAMMA = 1
    ALGEBRAIC_ALPHA = 1
    # ALGEBRAIC_GAMMA = 1


# ------------------------------------------------------------------------


class Gate(Layer):
    _label = GATE.LABEL
    _arrangement = GATE.ARRANGEMENT
    """
    A base gate layer that applies a linear or nonlinear function on the input (z) tensor to get an output (a) tensor.
    Arguments:
        size: gate size
        name: gate name
    """
    @MType(size=int, name=str)
    def __init__(self, *,
                 size=1,
                 name=''):
        self._z_t = None
        self._a_t = None
        self._monitor = None
        super().__init__(shape=(1, size), name=name)

    def __str__(self):
        return super().__str__() + '_' + GATE.LABEL

    # ------------------------------------------------------------------------

    @property
    def inputs(self):
        """
        Get gate forward pass input (z) tensor.
        Returns:
            tensor
        """
        if self._z_t is not None:
            return self._z_t.copy()
        else:
            return None

    @property
    def outputs(self):
        """
        Get gate forward pass output (a) tensor.
        Returns:
            tensor
        """
        if self._a_t is not None:
            return self._a_t.copy()
        else:
            return None

    def reset(self):
        """
        Reset internal states.
        """
        self._z_t = None
        self._a_t = None

    def unassign_hooks(self):
        """
        Unassign all callback or hook functions.
        """
        self._monitor = None

    @MType(monitor=OneOfType(callable, None))
    def assign_hook(self, *, monitor=None):
        """
        Assign callback or hook functions.
        Arguments:
            monitor: callback function to do probing during forward/backward pass
        """
        if monitor is not None:
            self._monitor = monitor

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return gate as a snapshot dict data
        Arguments:
            as_json:
            beautify_json:
        Returns:
            snapshot
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'base_label': Gate.label + '_' + snapshot['base_label']
        })
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()

# ------------------------------------------------------------------------


class Linear(Gate):
    _label = GATE.LINEAR_LABEL
    """
    A base linear gate layer where output (a) = input (z) tensor
    """

    # ------------------------------------------------------------------------

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return gate as a snapshot dict data
        Arguments:
            as_json:
            beautify_json:
        Returns:
            snapshot
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'base_label': Linear.label + '_' + snapshot['base_label']
        })
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()

    @MType(dict, np.ndarray, residue=dict)
    @MShape(axis=1)
    def forward(self, stage, z_t, *, residue={}):
        """
        Do forward pass by passing the input (z) tensor directly out as output (a) tensor.
        Arguments:
            stage: forward stage
            epoch: current training epoch
            z_t: input (z) tensor
            residue:
        Returns:
            gate
        """
        self._a_t = self._z_t = z_t  # z_t.copy()

        if self._monitor is not None:
            report = {
                'pass': 'forward',
                'stage': stage,
                'inputs': self.inputs,
                'outputs': self.outputs,
                'residue': residue
            }
            self._monitor(report)

        if self.has_next:
            return self.next.forward(stage, self._a_t, residue=residue)
        else:
            warnings.warn(
                'Linear gate {name} connection is incomplete. Missing connection to next layer.'.format(name=self.name),
                UserWarning)
            return self

    @MType(dict, np.ndarray, residue=dict)
    @MShape(axis=1)
    def backward(self, stage, eag_t, *, residue={}):
        """
        Do backward backward pass by passing 1 as gradient tensor of the input tensor.
        Arguments:
            stage: backward stage
            eag_t: error gradient w.r.t. post-nonlinear (a) tensor
            residue:
        Returns:
            gate
        """
        if self._a_t is None:
            warnings.warn(
                'Linear gate {name} cannot do backward pass. Need to run forward pass first.'.format(name=self.name),
                UserWarning)
            return self
        else:
            azg_t = Constant(1)(self._z_t.shape, dtype=np.float32)
            if self._monitor is not None:
                report = {
                    'pass': 'backward',
                    'stage': stage,
                    'grad': {
                        'az': azg_t,
                        'ea': eag_t
                    },
                    'residue': residue
                }
                self._monitor(report)

            if self.has_prev:
                return self.prev.backward(stage, azg_t, eag_t, residue=residue)
            else:
                self

# ------------------------------------------------------------------------


class Nonlinear(Gate):
    _label = GATE.NONLINEAR_LABEL
    """
    A base nonlinear gate layer. Manages and computes nonlinearity output from input tensor
    """

    # ------------------------------------------------------------------------

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return gate as a snapshot dict data
        Arguments:
            as_json:
            beautify_json:
        Returns:
            snapshot
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'base_label': Nonlinear.label + '_' + snapshot['base_label']
        })
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()

    @MType(dict, np.ndarray, residue=dict)
    @MShape(axis=1)
    def forward(self, stage, z_t, *, residue={}):
        """
        Do forward pass by calculating nonlinear tensor of the input (z) tensor
        Arguments:
            stage: forward stage
            z_t: pre-nonlinear (z) tensor
            residue:
        Returns:
            tail
        """
        self._z_t = z_t  # z_t.copy()

        self._a_t = self.compute_nonlinearity(self._z_t)

        if self._monitor is not None:
            report = {
                'pass': 'forward',
                'stage': stage,
                'inputs': self.inputs,
                'outputs': self.outputs,
                'residue': residue
            }
            self._monitor(report)

        if self.has_next:
            return self.next.forward(stage, self._a_t, residue=residue)
        else:
            warnings.warn(
                'Nonlinear gate {name} connection is incomplete. Missing connection to next layer.'.format(name=self.name),
                UserWarning)
            return self

    @MType(dict, np.ndarray, residue=dict)
    @MShape(axis=1)
    def backward(self, stage, eag_t, *, residue={}):
        """
        Do backward backward pass by calculating gradient of the forward pass input (z) tensor
        Arguments:
            stage: backward stage
            eag_t: error gradient w.r.t. post-nonlinear (a) tensor
            residue:
        Returns:
            head
        """
        if self._a_t is None:
            warnings.warn(
                'Nonlinear gate {name} cannot do backward pass. Need to run forward pass first.'.format(name=self.name),
                UserWarning)
            return self
        else:
            azg_t = self.compute_nonlinearity_grad(self._z_t, self._a_t)

            if self._monitor is not None:
                report = {
                    'pass': 'backward',
                    'stage': stage,
                    'grad': {
                        'az': azg_t,
                        'ea': eag_t
                    },
                    'residue': residue
                }
                self._monitor(report)

            if self.has_prev:
                return self.prev.backward(stage, azg_t, eag_t, residue=residue)
            else:
                return self

    @abc.abstractmethod
    def compute_nonlinearity(self):
        """
        Compute the nonlinear function to get post-nonlinear (a) tensor from pre-nonlinear (z) tensor.
        """
        pass

    @abc.abstractmethod
    def compute_nonlinearity_grad(self):
        """
        Compute the nonlinear function to get gradient post-nonlinear (a) tensor w.r.t. pre-nonlinear (z) tensor.
        """
        pass


# ------------------------------------------------------------------------


class ReLU(Nonlinear):
    _label = GATE.RELU_LABEL

    @MType(size=int, name=str)
    def __init__(self, *,
                 size=1,
                 name=''):
        self._cache = None
        super().__init__(size=size, name=name)

    # ------------------------------------------------------------------------

    @MType(np.ndarray)
    def compute_nonlinearity(self, z_t):
        """
        Compute the nonlinear function to get post-nonlinearity (a) tensor from pre-nonlinearity (z) tensor.
        Arguments:
            z_t: pre-nonlinearity (z) tensor
        Returns:
            tuple
        """
        pos_indices = z_t > 0

        a_t = np.multiply(z_t, pos_indices)

        self._cache = pos_indices

        return a_t

    @MType(np.ndarray, np.ndarray)
    def compute_nonlinearity_grad(self, z_t, a_t):
        """
        Compute the nonlinear function to get gradient post-nonlinearity (a) tensor w.r.t. pre-nonlinearity (z) tensor.
        Arguments:
            z_t: pre-nonlinearity (z) tensor
            a_t: post-nonlinearity (a) tensor
        Returns:
            tuple
        """
        pos_indices = self._cache

        azg_t = np.multiply(1, pos_indices)

        return azg_t


# ------------------------------------------------------------------------


class LeakyReLU(Nonlinear):
    _label = GATE.LEAKY_RELU_LABEL

    @MType(size=int, name=str)
    def __init__(self, *,
                 size=1,
                 name=''):
        self._cache = None
        super().__init__(size=size, name=name)

    # ------------------------------------------------------------------------

    def reset(self):
        """
        Reset internal evaluation states
        """
        super().reset()
        self._cache = None

    @MType(np.ndarray)
    def compute_nonlinearity(self, z_t):
        """
        Compute the nonlinear function to get post-nonlinearity (a) tensor from pre-nonlinearity (z) tensor.
        Arguments:
            z_t: pre-nonlinearity (z) tensor
        Returns:
            tuple
        """
        neg_indices = z_t <= 0

        a_t = np.where(neg_indices, GATE.LEAKY_RELU_ALPHA * z_t, z_t)

        self._cache = neg_indices

        return a_t

    @MType(np.ndarray, np.ndarray)
    def compute_nonlinearity_grad(self, z_t, a_t):
        """
        Compute the nonlinear function to get gradient post-nonlinearity (a) tensor w.r.t. pre-nonlinearity (z) tensor.
        Arguments:
            z_t: pre-nonlinearity (z) tensor
            a_t: post-nonlinearity (a) tensor
        Returns:
            tuple
        """

        neg_indices = self._cache

        azg_t = np.ones_like(a_t, dtype=a_t.dtype)
        azg_t[neg_indices] = GATE.LEAKY_RELU_ALPHA

        return azg_t


# ------------------------------------------------------------------------


class ELU(Nonlinear):
    _label = GATE.ELU_LABEL

    @MType(size=int, name=str)
    def __init__(self, *,
                 size=1,
                 name=''):
        self._cache = None
        super().__init__(size=size, name=name)

    # ------------------------------------------------------------------------

    def reset(self):
        """
        Reset internal evaluation states
        """
        super().reset()
        self._cache = None

    @MType(np.ndarray)
    def compute_nonlinearity(self, z_t):
        """
        Compute the nonlinear function to get post-nonlinearity (a) tensor from pre-nonlinearity (z) tensor.
        Arguments:
            z_t: pre-nonlinearity (z) tensor
        Returns:
            tuple
        """
        neg_indices = z_t <= 0

        a_t = z_t.copy()
        a_t[neg_indices] = GATE.ELU_ALPHA * (np.exp(z_t[neg_indices]) - 1)

        self._cache = neg_indices

        return a_t

    @MType(np.ndarray, np.ndarray)
    def compute_nonlinearity_grad(self, z_t, a_t):
        """
        Compute the nonlinear function to get gradient post-nonlinearity (a) tensor w.r.t. pre-nonlinearity (z) tensor.
        Arguments:
            z_t: pre-nonlinearity (z) tensor
            a_t: post-nonlinearity (a) tensor
        Returns:
            tuple
        """
        neg_indices = self._cache

        azg_t = np.ones_like(a_t, dtype=a_t.dtype)
        azg_t[neg_indices] = a_t[neg_indices] + GATE.ELU_ALPHA

        return azg_t


# ------------------------------------------------------------------------


class Swish(Nonlinear):
    _label = GATE.SWISH_LABEL

    @MType(size=int, name=str)
    def __init__(self, *,
                 size=1,
                 name=''):
        self._cache = None
        super().__init__(size=size, name=name)

    # ------------------------------------------------------------------------

    def reset(self):
        """
        Reset internal evaluation states
        """
        super().reset()
        self._cache = None

    @MType(np.ndarray)
    def compute_nonlinearity(self, z_t):
        """
        Compute the nonlinear function to get post-nonlinearity (a) tensor from pre-nonlinearity (z) tensor.
        Arguments:
            z_t: pre-nonlinearity (z) tensor
        Returns:
            tuple
        """
        exp_of_z_t = np.exp(-GATE.SWISH_ALPHA * z_t + 1e-12)
        inv_exp_of_z_t = 1 / (1 + exp_of_z_t + GATE.SWISH_BETA * z_t)

        a_t = np.multiply(z_t, inv_exp_of_z_t) + GATE.SWISH_GAMMA * z_t

        self._cache = (exp_of_z_t, inv_exp_of_z_t)

        # sigmoid_of_z_t = np.exp(-np.logaddexp(0, -GATE.SWISH_ALPHA * z_t))
        #
        # a_t = np.multiply(z_t, sigmoid_of_z_t)
        #
        # self._cache = sigmoid_of_z_t

        return a_t

    @MType(np.ndarray, np.ndarray)
    def compute_nonlinearity_grad(self, z_t, a_t):
        """
        Compute the nonlinear function to get gradient post-nonlinearity (a) tensor w.r.t. pre-nonlinearity (z) tensor.
        Arguments:
            z_t: pre-nonlinearity (z) tensor
            a_t: post-nonlinearity (a) tensor
        Returns:
            tuple
        """
        (exp_of_z_t, inv_exp_of_z_t) = self._cache
        azg_t = inv_exp_of_z_t - np.multiply(np.multiply(z_t, GATE.SWISH_BETA - GATE.SWISH_ALPHA * exp_of_z_t), np.square(inv_exp_of_z_t)) + GATE.SWISH_GAMMA

        # sigmoid_of_z_t = self._cache
        # azg_t = sigmoid_of_z_t + np.multiply(np.multiply(GATE.SWISH_ALPHA * z_t, sigmoid_of_z_t), (1 - sigmoid_of_z_t))

        return azg_t


# ------------------------------------------------------------------------


class SoftPlus(Nonlinear):
    _label = GATE.SOFT_PLUS_LABEL

    @MType(np.ndarray)
    def compute_nonlinearity(self, z_t):
        """
        Compute the nonlinear function to get post-nonlinearity (a) tensor from pre-nonlinearity (z) tensor.
        Arguments:
            z_t: pre-nonlinearity (z) tensor
        Returns:
            tuple
        """
        # a_t = GATE.SOFT_PLUS_GAMMA * np.log(1 + np.exp(GATE.SOFT_PLUS_ALPHA * z_t)) + GATE.SOFT_PLUS_BETA * z_t
        a_t = GATE.SOFT_PLUS_GAMMA * (np.log(1 + np.exp(-np.abs(GATE.SOFT_PLUS_ALPHA * z_t))) + np.maximum(GATE.SOFT_PLUS_ALPHA * z_t, 0)) + GATE.SOFT_PLUS_BETA * z_t

        return a_t

    @MType(np.ndarray, np.ndarray)
    def compute_nonlinearity_grad(self, z_t, a_t):
        """
        Compute the nonlinear function to get gradient post-nonlinearity (a) tensor w.r.t. pre-nonlinearity (z) tensor.
        Arguments:
            z_t: pre-nonlinearity (z) tensor
            a_t: post-nonlinearity (a) tensor
        Returns:
            tuple
        """
        azg_t = GATE.SOFT_PLUS_GAMMA * GATE.SOFT_PLUS_ALPHA * np.exp(-np.logaddexp(0, -GATE.SOFT_PLUS_ALPHA * z_t)) + GATE.SOFT_PLUS_BETA

        return azg_t


# ------------------------------------------------------------------------


class Sigmoid(Nonlinear):
    _label = GATE.SIGMOID_LABEL

    # ------------------------------------------------------------------------

    @MType(np.ndarray)
    def compute_nonlinearity(self, z_t):
        """
        Compute the nonlinear function to get post-nonlinearity (a) tensor from pre-nonlinearity (z) tensor.
        Arguments:
            z_t: pre-nonlinearity (z) tensor
        Returns:
            tuple
        """
        # a_t = np.exp(-np.logaddexp(0, -GATE.SIGMOID_ALPHA * z_t))
        exp_of_z_t = np.exp(GATE.SIGMOID_ALPHA * z_t)
        a_t = GATE.SIGMOID_GAMMA * exp_of_z_t / (1 + GATE.SIGMOID_GAMMA * exp_of_z_t)
        return a_t

    @MType(np.ndarray, np.ndarray)
    def compute_nonlinearity_grad(self, z_t, a_t):
        """
        Compute the nonlinear function to get gradient post-nonlinearity (a) tensor w.r.t. pre-nonlinearity (z) tensor.
        Arguments:
            z_t: pre-nonlinearity (z) tensor
            a_t: post-nonlinearity (a) tensor
        Returns:
            tuple
        """
        azg_t = np.multiply(GATE.SIGMOID_ALPHA * a_t, (1 - a_t))
        return azg_t


# ------------------------------------------------------------------------


class Tanh(Nonlinear):
    _label = GATE.TANH_LABEL

    # ------------------------------------------------------------------------

    @MType(np.ndarray)
    def compute_nonlinearity(self, z_t):
        """
        Compute the nonlinear function to get post-nonlinearity (a) tensor from pre-nonlinearity (z) tensor.
        Arguments:
            z_t: pre-nonlinearity (z) tensor
        Returns:
            tuple
        """
        a_t = np.tanh(GATE.TANH_ALPHA * z_t)
        return a_t

    @MType(np.ndarray, np.ndarray)
    def compute_nonlinearity_grad(self, z_t, a_t):
        """
        Compute the nonlinear function to get gradient post-nonlinearity (a) tensor w.r.t. pre-nonlinearity (z) tensor.
        Arguments:
            z_t: pre-nonlinearity (z) tensor
            a_t: post-nonlinearity (a) tensor
        Returns:
            tuple
        """
        azg_t = GATE.TANH_ALPHA * (1 - np.square(GATE.TANH_ALPHA * a_t))
        return azg_t

# ------------------------------------------------------------------------


class Algebraic(Nonlinear):
    _label = GATE.ALGEBRAIC_LABEL

    @MType(size=int, name=str)
    def __init__(self, *,
                 size=1,
                 name=''):
        self._cache = None
        super().__init__(size=size, name=name)

    # ------------------------------------------------------------------------

    def reset(self):
        """
        Reset internal evaluation states
        """
        super().reset()
        self._cache = None

    @MType(np.ndarray)
    def compute_nonlinearity(self, z_t):
        """
        Compute the nonlinear function to get post-nonlinearity (a) tensor from pre-nonlinearity (z) tensor.
        Arguments:
            z_t: pre-nonlinearity (z) tensor
        Returns:
            tuple
        """
        inv_of_z_t = 1 / (1 + GATE.ALGEBRAIC_ALPHA * np.square(z_t))
        inv_sqrt_of_z_t = GATE.ALGEBRAIC_ALPHA * np.sqrt(inv_of_z_t)
        a_t = np.multiply(z_t, inv_sqrt_of_z_t)

        self._cache = (inv_of_z_t, inv_sqrt_of_z_t)

        return a_t

    @MType(np.ndarray, np.ndarray)
    def compute_nonlinearity_grad(self, z_t, a_t):
        """
        Compute the nonlinear function to get gradient post-nonlinearity (a) tensor w.r.t. pre-nonlinearity (z) tensor.
        Arguments:
            z_t: pre-nonlinearity (z) tensor
            a_t: post-nonlinearity (a) tensor
        Returns:
            tuple
        """
        (inv_of_z_t, inv_sqrt_of_z_t) = self._cache
        azg_t = np.multiply(inv_of_z_t, inv_sqrt_of_z_t)
        return azg_t
