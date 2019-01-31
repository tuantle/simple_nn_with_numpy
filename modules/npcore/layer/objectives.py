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
import copy
import warnings
import json
import numpy as np

from util.const import CONST
from util.validation import (
    MShape,
    MType,
    OneOfType
)
from npcore.layer.layer import Layer

# ------------------------------------------------------------------------


class OBJECTIVE(CONST):
    LABEL = 'objective'
    MAE_LOSS_LABEL = 'mae_loss'
    MSE_LOSS_LABEL = 'mse_loss'
    LOG_COSH_LOSS_LABEL = 'log_cosh_loss'
    XTANH_LOSS_LABEL = 'xtanh_loss'
    XSIGMOID_LOSS_LABEL = 'xsigmoid_loss'
    ALGEBRAIC_LOSS_LABEL = 'algebraic_loss'
    SIGMOID_CROSSENTROPY_LOSS = 'sigmoid_crossentropy_loss'
    SOFTMAX_CROSSENTROPY_LOSS = 'softmax_crossentropy_loss'
    ARRANGEMENT = ('2', '')

# ------------------------------------------------------------------------


class Objective(Layer):
    _label = OBJECTIVE.LABEL
    _arrangement = OBJECTIVE.ARRANGEMENT
    """
    Abtraction of a base objective layer. Manages objective loss.
    Arguments:
        size: objective size
        name: objective name
        metric: loss metric
    """
    @MType(size=int,
           name=str,
           metric=(str,))
    def __init__(self, *,
                 size=1,
                 name='',
                 metric=('loss',)):
        self._y_t = None
        self._y_prime_t = None
        self._evaluation = {
            'count': 0,
            'metric': {}
        }
        self._residue = {}

        self._monitor = None

        super().__init__(shape=(1, size), name=name)
        self.reconfig(metric=metric)

    def __str__(self):
        return super().__str__() + '_' + OBJECTIVE.LABEL

    # ------------------------------------------------------------------------

    @property
    def inputs(self):
        """
        Get objective forward pass input tensor.
        Returns:
            tensor
        """
        if self.has_prev:
            return self.prev.outputs
        else:
            return None

    @property
    def outputs(self):
        """
        Get objective forward pass output tensor
        Returns:
            tensor
        """
        if self._y_t is not None:
            return self._y_t.copy()
        else:
            return None

    @property
    def evaluation_metric(self):
        """
        Get objective evaluation metric
        """
        evaluation_count = self._evaluation['count']
        evaluation_metric = copy.deepcopy(self._evaluation['metric'])

        if evaluation_count > 1:
            for key in evaluation_metric.keys():
                evaluation_metric[key] /= evaluation_count
        return evaluation_metric

    def unassign_hooks(self):
        """
        Unassign all callback functions
        """
        self._monitor = None

    @MType(monitor=OneOfType(callable, None))
    def assign_hook(self, *,
                    monitor=None):
        """
        Assign callback functions
        Arguments:
            monitor: callback function to do probing during forward/backward pass
        """
        if monitor is not None:
            self._monitor = monitor

    def reset(self):
        """
        Reset internal states.
        """
        self._y_t = None
        self._y_prime_t = None

        self._residue = {}
        self._evaluation['count'] = 0
        for key in self._evaluation['metric'].keys():
            self._evaluation['metric'][key] = 0

    @MType(shape=OneOfType((int,), None),
           metric=OneOfType((str,), None))
    def reconfig(self, *,
                 shape=None,
                 metric=None):
        """
        Reconfig objective
        Arguments:
            shape: objective layer shape
            metric: loss metric
        """
        if metric is not None:
            if 'loss' in metric:
                self._evaluation['metric']['loss'] = 0
            else:
                raise TypeError('Unknown metric {metric} for objective {name}.'.format(metric=metric, name=self.name))
        if shape is not None:
            super().reconfig(shape=shape)
        self.reset()

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return objective as a snapshot dict data
        Arguments:
            as_json:
            beautify_json:
        Returns:
            snapshot
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'base_label': Objective.label + '_' + snapshot['base_label'],
            'metric': tuple(self._evaluation['metric'].keys())
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
    def forward(self, stage, a_t, *, residue={}):
        """
        Do forward pass method.
        Arguments:
            stage: forward stage
            a_t: post-nonlinearity (a) tensor
            residue:
        Returns:
            layer
        """
        self._y_t = a_t  # a_t.copy()
        self._residue = residue

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
            warnings.warn(
                'Objective {name} layer must be the last in connection. There should be no connection to next layer.'.format(name=self.name),
                UserWarning)
        return self

    @MType(np.ndarray)
    @MShape(axis=1)
    def evaluate(self, y_prime_t):
        """
        Get evaluation metric given the expected truth.
        Arguments:
            y_prime_t: expected output (y) tensor
        Returns:
            self
        """
        self._evaluation['count'] += 1
        self._y_prime_t = y_prime_t  # y_prime_t.copy()
        evaluation_metric = self._evaluation['metric']

        (ly_t, residue) = self.compute_loss(self._y_t, self._y_prime_t, residue=self._residue)
        metric = self.compute_evaluation_metric(self._y_t, self._y_prime_t, ly_t, evaluation_metric)

        self._evaluation['metric'] = metric
        self._residue = residue

        return self

    @MType(dict)
    def backward(self, stage):
        """
        Do backward pass by passing the loss gradient tensor back to the prev link.
        Arguments:
            stage: backward stage
        Returns:
            layer
        """
        if self._y_t is None:
            warnings.warn(
                'Objective {name} cannot do backward pass. Need to run forward pass first.'.format(name=self.name),
                UserWarning)
            return self
        elif self._y_prime_t is None:
            warnings.warn(
                'Objective {name} cannot do backward pass. Need to run evaluation first.'.format(name=self.name),
                UserWarning)
            return self
        else:
            hparam = stage['hparam']
            batch_size = hparam['batch_size']

            (eyg_t, residue) = self.compute_loss_grad(self._y_t, self._y_prime_t, residue=self._residue)
            eyg_t = eyg_t / batch_size if batch_size > 1 else eyg_t

            if self._monitor is not None:
                report = {
                    'pass': 'backward',
                    'stage': stage,
                    'error': self._ey_t,
                    'grad': {
                        'error': eyg_t
                    },
                    'evaluation': self._evaluation,
                    'residue': residue
                }
                self._monitor(report)

            if self.has_prev:
                return self.prev.backward(stage, eyg_t, residue=residue)
            else:
                warnings.warn(
                    'Objective {name} connection is incomplete. Missing connection to previous layer.'.format(name=self.name),
                    UserWarning)
                return self

    @abc.abstractmethod
    def compute_evaluation_metric(self):
        """
        Compute the evaluation metric.
        """
        pass

    @abc.abstractmethod
    def compute_loss(self):
        """
        Compute the loss tensor. Not implemented
        """
        pass

    @abc.abstractmethod
    def compute_loss_grad(self):
        """
        Compute the loss gradient tensor for backpropagation. Not implemented
        """
        pass


# ------------------------------------------------------------------------


class MAELoss(Objective):
    _label = OBJECTIVE.MAE_LOSS_LABEL
    """
    Objective using mean absolute error for loss function
    """

    # ------------------------------------------------------------------------
    @MType(shape=OneOfType((int,), None),
           metric=OneOfType((str,), None))
    def reconfig(self, *,
                 shape=None,
                 metric=None):
        """
        Reconfig objective
        Arguments:
            shape: objective layer shape
            metric: loss metric
        """
        if metric is not None:
            if 'loss' in metric or ('accuracy' or 'acc') in metric:
                if 'loss' in metric:
                    self._evaluation['metric']['loss'] = 0
            if ('accuracy' or 'acc') in metric or \
               ('recall' or 'rc') in metric or \
               ('precision' or 'prec') in metric or \
               ('f1_score' or 'f1') in metric:
                warnings.warn(
                    'Mean absolute error objective only have loss metric. Ignoring metrics {metric}'.format(metric=metric),
                    UserWarning)
            else:
                raise TypeError('Unknown metric {metric} for objective {name}.'.format(metric=metric, name=self.name))
        if shape is not None:
            super().reconfig(shape=shape)
        self.reset()

    @MType(np.ndarray, np.ndarray, dict)
    def compute_loss(self, y_t, y_prime_t, *, residue={}):
        """
        Compute the loss.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            residue:
        Returns:
            tuple
        """
        ey_t = y_t - y_prime_t
        ly_t = np.abs(ey_t)
        return (ly_t, residue)

    @MType(np.ndarray, np.ndarray, dict)
    def compute_loss_grad(self, y_t, y_prime_t, *, residue={}):
        """
        Compute the loss gradient tensor for gradient descent update.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            residue:
        Returns:
            tuple
        """
        eyg_t = np.vectorize(lambda element: (element and 1) or (not element and -1))(y_t > y_prime_t)

        return (eyg_t, residue)

    @MType(np.ndarray, np.ndarray, np.ndarray, dict)
    def compute_evaluation_metric(self, y_t, y_prime_t, ly_t, evaluation_metric):
        """
        Compute the evaluation metric.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            ly_t: loss tensor
        Returns:
            metric
        """
        if 'loss' in evaluation_metric:
            evaluation_metric['loss'] += ly_t.mean()

        return evaluation_metric

# ------------------------------------------------------------------------


class MSELoss(Objective):
    _label = OBJECTIVE.MSE_LOSS_LABEL
    """
    Objective using mean square error for loss function.
    """

    # ------------------------------------------------------------------------

    @MType(shape=OneOfType((int,), None),
           metric=OneOfType((str,), None))
    def reconfig(self, *,
                 shape=None,
                 metric=None):
        """
        Reconfig objective
        Arguments:
            shape: objective layer shape
            metric: loss metric
        """
        if metric is not None:
            if 'loss' in metric:
                self._evaluation['metric']['loss'] = 0
            if ('accuracy' or 'acc') in metric or \
               ('recall' or 'rc') in metric or \
               ('precision' or 'prec') in metric or \
               ('f1_score' or 'f1') in metric:
                warnings.warn(
                    'Mean square error objective only have loss metric. Ignoring metrics {metric}'.format(metric=metric),
                    UserWarning)
            else:
                raise TypeError('Unknown metric {metric} for objective {name}.'.format(metric=metric, name=self.name))
        if shape is not None:
            super().reconfig(shape=shape)
        self.reset()

    @MType(np.ndarray, np.ndarray, dict)
    def compute_loss(self, y_t, y_prime_t, *, residue={}):
        """
        Compute the loss.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            residue:
        Returns:
            tuple
        """
        ey_t = y_t - y_prime_t
        ly_t = np.square(ey_t)
        return (ly_t, residue)

    @MType(np.ndarray, np.ndarray, dict)
    def compute_loss_grad(self, y_t, y_prime_t, *, residue={}):
        """
        Compute the loss gradient tensor for gradient descent update.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            residue:
        Returns:
            tuple
        """
        ey_t = y_t - y_prime_t
        eyg_t = 2 * ey_t

        return (eyg_t, residue)

    @MType(np.ndarray, np.ndarray, np.ndarray, dict)
    def compute_evaluation_metric(self, y_t, y_prime_t, ly_t, evaluation_metric):
        """
        Compute the evaluation metric.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            ly_t: loss tensor
        Returns:
            metric
        """
        if 'loss' in evaluation_metric:
            evaluation_metric['loss'] += ly_t.mean()

        return evaluation_metric


# ------------------------------------------------------------------------


class LogCoshLoss(Objective):
    _label = OBJECTIVE.LOG_COSH_LOSS_LABEL
    """
    Objective using log-cosh loss for loss functionself.
    `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
    to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
    like the l2 loss, but will not be so strongly affected by the
    occasional wildly incorrect prediction.
    """

    # ------------------------------------------------------------------------

    @MType(shape=OneOfType((int,), None),
           metric=OneOfType((str,), None))
    def reconfig(self, *,
                 shape=None,
                 metric=None):
        """
        Reconfig objective
        Arguments:
            shape: objective layer shape
            metric: loss metric
        """
        if metric is not None:
            if 'loss' in metric or ('accuracy' or 'acc') in metric:
                if 'loss' in metric:
                    self._evaluation['metric']['loss'] = 0
                if ('accuracy' or 'acc') in metric or \
                   ('recall' or 'rc') in metric or \
                   ('precision' or 'prec') in metric or \
                   ('f1_score' or 'f1') in metric:
                    warnings.warn(
                        'Log-cosh loss objective only have loss metric. Ignoring metrics {metric}'.format(metric=metric),
                        UserWarning)
            else:
                raise TypeError('Unknown metric {metric} for objective {name}.'.format(metric=metric, name=self.name))
        if shape is not None:
            super().reconfig(shape=shape)
        self.reset()

    @MType(np.ndarray, np.ndarray, dict)
    def compute_loss(self, y_t, y_prime_t, *, residue={}):
        """
        Compute the loss.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            residue:
        Returns:
            tuple
        """
        ey_t = y_t - y_prime_t
        ly_t = np.log(np.cosh(ey_t) + 1e-12)

        return (ly_t, residue)

    @MType(np.ndarray, np.ndarray, dict)
    def compute_loss_grad(self, y_t, y_prime_t, *, residue={}):
        """
        Compute the loss gradient tensor for gradient descent update.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            residue:
        Returns:
            tuple
        """
        ey_t = y_t - y_prime_t
        eyg_t = np.tanh(ey_t)

        return (eyg_t, residue)

    @MType(np.ndarray, np.ndarray, np.ndarray, dict)
    def compute_evaluation_metric(self, y_t, y_prime_t, ly_t, evaluation_metric):
        """
        Compute the evaluation metric.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            ly_t: loss tensor
        Returns:
            metric
        """
        if 'loss' in evaluation_metric:
            evaluation_metric['loss'] += ly_t.mean()

        return evaluation_metric


# ------------------------------------------------------------------------


class XTanhLoss(Objective):
    _label = OBJECTIVE.XTANH_LOSS_LABEL
    """
    Arguments:
        size: objective size
        name: objective name
        metric: loss metric
    """
    @MType(size=int,
           name=str,
           metric=(str,))
    def __init__(self, *,
                 size=1,
                 name='',
                 metric=('loss',)):
        self._cache = None
        super().__init__(size=size, name=name)
        self.reconfig(metric=metric)

    # ------------------------------------------------------------------------

    @MType(shape=OneOfType((int,), None),
           metric=OneOfType((str,), None))
    def reconfig(self, *,
                 shape=None,
                 metric=None):
        """
        Reconfig objective
        Arguments:
            shape: objective layer shape
            metric: loss metric
        """
        if metric is not None:
            if 'loss' in metric or ('accuracy' or 'acc') in metric:
                if 'loss' in metric:
                    self._evaluation['metric']['loss'] = 0
                if ('accuracy' or 'acc') in metric or \
                   ('recall' or 'rc') in metric or \
                   ('precision' or 'prec') in metric or \
                   ('f1_score' or 'f1') in metric:
                    warnings.warn(
                        'XTanh loss objective only have loss metric. Ignoring metrics {metric}'.format(metric=metric),
                        UserWarning)
            else:
                raise TypeError('Unknown metric {metric} for objective {name}.'.format(metric=metric, name=self.name))
        if shape is not None:
            super().reconfig(shape=shape)
        self.reset()

    @MType(np.ndarray, np.ndarray, dict)
    def compute_loss(self, y_t, y_prime_t, *, residue={}):
        """
        Compute the loss.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            residue:
        Returns:
            tuple
        """
        ey_t = y_t - y_prime_t
        tanh_of_ey_t = np.tanh(ey_t)
        ly_t = np.multiply(ey_t, tanh_of_ey_t)
        self._cache = tanh_of_ey_t

        return (ly_t, residue)

    @MType(np.ndarray, np.ndarray, dict)
    def compute_loss_grad(self, y_t, y_prime_t, *, residue={}):
        """
        Compute the loss gradient tensor for gradient descent update.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            residue:
        Returns:
            tuple
        """
        ey_t = y_t - y_prime_t
        tanh_of_ey_t = self._cache
        eyg_t = tanh_of_ey_t + ey_t * (1 - np.square(tanh_of_ey_t))

        return (eyg_t, residue)

    @MType(np.ndarray, np.ndarray, np.ndarray, dict)
    def compute_evaluation_metric(self, y_t, y_prime_t, ly_t, evaluation_metric):
        """
        Compute the evaluation metric.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            ly_t: loss tensor
        Returns:
            metric
        """
        if 'loss' in evaluation_metric:
            evaluation_metric['loss'] += ly_t.mean()

        return evaluation_metric


# ------------------------------------------------------------------------


class XSigmoidLoss(Objective):
    _label = OBJECTIVE.XSIGMOID_LOSS_LABEL
    """
    Arguments:
        size: objective size
        name: objective name
        metric: loss metric
    """
    @MType(size=int,
           name=str,
           metric=(str,))
    def __init__(self, *,
                 size=1,
                 name='',
                 metric=('loss',)):
        self._cache = None
        super().__init__(size=size, name=name)
        self.reconfig(metric=metric)

    # ------------------------------------------------------------------------

    @MType(shape=OneOfType((int,), None),
           metric=OneOfType((str,), None))
    def reconfig(self, *,
                 shape=None,
                 metric=None):
        """
        Reconfig objective
        Arguments:
            shape: objective layer shape
            metric: loss metric
        """
        if metric is not None:
            if 'loss' in metric or ('accuracy' or 'acc') in metric:
                if 'loss' in metric:
                    self._evaluation['metric']['loss'] = 0
                if ('accuracy' or 'acc') in metric or \
                   ('recall' or 'rc') in metric or \
                   ('precision' or 'prec') in metric or \
                   ('f1_score' or 'f1') in metric:
                    warnings.warn(
                        'XSigmoid loss objective only have loss metric. Ignoring metrics {metric}'.format(metric=metric),
                        UserWarning)
            else:
                raise TypeError('Unknown metric {metric} for objective {name}.'.format(metric=metric, name=self.name))
        if shape is not None:
            super().reconfig(shape=shape)
        self.reset()

    @MType(np.ndarray, np.ndarray, dict)
    def compute_loss(self, y_t, y_prime_t, *, residue={}):
        """
        Compute the loss.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            residue:
        Returns:
            tuple
        """
        ey_t = y_t - y_prime_t
        sigmoid_of_ey_t = np.exp(-np.logaddexp(0, -ey_t + 1e-12))

        ly_t = np.multiply(2 * ey_t, sigmoid_of_ey_t) - ey_t
        self._cache = sigmoid_of_ey_t

        return (ly_t, residue)

    @MType(np.ndarray, np.ndarray, dict)
    def compute_loss_grad(self, y_t, y_prime_t, *, residue={}):
        """
        Compute the loss gradient tensor for gradient descent update.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            residue:
        Returns:
            tuple
        """
        ey_t = y_t - y_prime_t
        sigmoid_of_ey_t = self._cache
        eyg_t = 2 * sigmoid_of_ey_t + np.multiply(np.multiply(2 * ey_t, np.exp(-ey_t)), np.square(sigmoid_of_ey_t)) - 1

        return (eyg_t, residue)

    @MType(np.ndarray, np.ndarray, np.ndarray, dict)
    def compute_evaluation_metric(self, y_t, y_prime_t, ly_t, evaluation_metric):
        """
        Compute the evaluation metric.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            ly_t: loss tensor
        Returns:
            metric
        """
        if 'loss' in evaluation_metric:
            evaluation_metric['loss'] += ly_t.mean()

        return evaluation_metric


# ------------------------------------------------------------------------


class AlgebraicLoss(Objective):
    _label = OBJECTIVE.ALGEBRAIC_LOSS_LABEL
    """
    Arguments:
        size: objective size
        name: objective name
        metric: loss metric
    """
    @MType(size=int,
           name=str,
           metric=(str,))
    def __init__(self, *,
                 size=1,
                 name='',
                 metric=('loss',)):
        self._cache = None
        super().__init__(size=size, name=name)
        self.reconfig(metric=metric)

    # ------------------------------------------------------------------------

    @MType(shape=OneOfType((int,), None),
           metric=OneOfType((str,), None))
    def reconfig(self, *,
                 shape=None,
                 metric=None):
        """
        Reconfig objective
        Arguments:
            shape: objective layer shape
            metric: loss metric
        """
        if metric is not None:
            if 'loss' in metric or ('accuracy' or 'acc') in metric:
                if 'loss' in metric:
                    self._evaluation['metric']['loss'] = 0
                if ('accuracy' or 'acc') in metric or \
                   ('recall' or 'rc') in metric or \
                   ('precision' or 'prec') in metric or \
                   ('f1_score' or 'f1') in metric:
                    warnings.warn(
                        'Algebraic loss objective only have loss metric. Ignoring metrics {metric}'.format(metric=metric),
                        UserWarning)
            else:
                raise TypeError('Unknown metric {metric} for objective {name}.'.format(metric=metric, name=self.name))
        if shape is not None:
            super().reconfig(shape=shape)
        self.reset()

    @MType(np.ndarray, np.ndarray, dict)
    def compute_loss(self, y_t, y_prime_t, *, residue={}):
        """
        Compute the loss.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            residue:
        Returns:
            tuple
        """
        ey_t = y_t - y_prime_t
        sqr_of_ey_t = np.square(ey_t)
        inv_of_ey_t = 1 / (1 + sqr_of_ey_t)
        inv_sqrt_of_ey_t = np.sqrt(inv_of_ey_t)
        ly_t = np.multiply(sqr_of_ey_t, inv_sqrt_of_ey_t)
        self._cache = (sqr_of_ey_t, inv_of_ey_t, inv_sqrt_of_ey_t)

        return (ly_t, residue)

    @MType(np.ndarray, np.ndarray, dict)
    def compute_loss_grad(self, y_t, y_prime_t, *, residue={}):
        """
        Compute the loss gradient tensor for gradient descent update.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            residue:
        Returns:
            tuple
        """
        ey_t = y_t - y_prime_t
        (sqr_of_ey_t, inv_of_ey_t, inv_sqrt_of_ey_t) = self._cache
        eyg_t = np.multiply(2 * ey_t + np.multiply(ey_t, sqr_of_ey_t), np.multiply(inv_of_ey_t, inv_sqrt_of_ey_t))

        return (eyg_t, residue)

    @MType(np.ndarray, np.ndarray, np.ndarray, dict)
    def compute_evaluation_metric(self, y_t, y_prime_t, ly_t, evaluation_metric):
        """
        Compute the evaluation metric.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            ly_t: loss tensor
        Returns:
            metric
        """
        if 'loss' in evaluation_metric:
            evaluation_metric['loss'] += ly_t.mean()

        return evaluation_metric


# ------------------------------------------------------------------------


class SigmoidCrossentropyLoss(Objective):
    _label = OBJECTIVE.SIGMOID_CROSSENTROPY_LOSS
    """
    Objective using sigmoid (binary)crossentropyfor loss function.
    Arguments:
        size: objective size
        name: objective name
        metric: loss and accuracy metrics
    """
    @MType(size=int,
           name=str,
           metric=(str,))
    def __init__(self, *,
                 size=1,
                 name='',
                 metric=('loss', 'accuracy')):
        super().__init__(size=size, name=name)
        self.reconfig(metric=metric)

    # ------------------------------------------------------------------------

    @MType(shape=OneOfType((int,), None),
           metric=OneOfType((str,), None))
    def reconfig(self, *,
                 shape=None,
                 metric=None):
        """
        Reconfig objective
        Arguments:
            shape: objective layer shape
            metric: loss metric
        """
        if metric is not None:
            if 'loss' in metric or ('accuracy' or 'acc'):
                if 'loss' in metric:
                    self._evaluation['metric']['loss'] = 0
                if ('accuracy' or 'acc') in metric:
                    self._evaluation['metric']['accuracy'] = 0
                if ('recall' or 'rc') in metric:
                    self._evaluation['metric']['recall'] = 0
                if ('precision' or 'prec') in metric:
                    self._evaluation['metric']['precision'] = 0
                if ('f1_score' or 'f1') in metric:
                    self._evaluation['metric']['f1_score'] = 0
            else:
                raise TypeError('Unknown metric {metric} for objective {name}.'.format(metric=metric, name=self.name))
        if shape is not None:
            super().reconfig(shape=shape)
        self.reset()

    @MType(dict, np.ndarray, residue=dict)
    @MShape(axis=1)
    def forward(self, stage, a_t, *, residue={}):
        """
        Do forward pass method.
        Arguments:
            stage: forward stage
            a_t: post-nonlinearity (a) tensor
            residue:
        Returns:
            layer
        """
        sigmoid_of_a_t = np.exp(-np.logaddexp(0, -a_t + 1e-12))
        return super().forward(stage, sigmoid_of_a_t, residue=residue)

    @MType(np.ndarray, np.ndarray, dict)
    def compute_loss(self, y_t, y_prime_t, *, residue={}):
        """
        Compute the loss.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            residue:
        Returns:
            tuple
        """
        y_prime_t = y_prime_t.astype(np.float32)
        ly_t = -(y_prime_t * np.log(y_t + 1e-12) + (1 - y_prime_t) * np.log((1 - y_t) + 1e-12))

        return (ly_t, residue)

    @MType(np.ndarray, np.ndarray, dict)
    def compute_loss_grad(self, y_t, y_prime_t, *, residue={}):
        """
        Compute the loss gradient tensor for gradient descent update.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            residue:
        Returns:
            tuple
        """
        ey_t = y_t - y_prime_t
        eyg_t = ey_t

        return (eyg_t, residue)

    @MType(np.ndarray, np.ndarray, np.ndarray, dict)
    def compute_evaluation_metric(self, y_t, y_prime_t, ly_t, evaluation_metric):
        """
        Compute the evaluation metric.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            ly_t: loss tensor
        Returns:
            metric
        """
        if 'loss' in evaluation_metric:
            evaluation_metric['loss'] += ly_t.mean()
        if 'accuracy' in evaluation_metric:
            evaluation_metric['accuracy'] += np.equal(y_prime_t, y_t.round()).astype(np.int8).mean()
        if 'recall' in evaluation_metric or 'precision' in evaluation_metric or 'f1_score' in evaluation_metric:
            y_t = np.round(y_t)
            true_pos = np.sum(np.multiply(y_t, y_prime_t), axis=0).astype(np.float)
            # true_neg = np.sum(np.multiply((1 - y_t), (1 - y_prime_t)), axis=0).astype(np.float)
            false_pos = np.sum(np.multiply(y_t, (1 - y_prime_t)), axis=0).astype(np.float)
            false_neg = np.sum(np.multiply((1 - y_t), y_prime_t), axis=0).astype(np.float)
            recall = true_pos / (true_pos + false_neg + 1e-12)
            precision = true_pos / (true_pos + false_pos + 1e-12)
            if 'recall' in evaluation_metric:
                evaluation_metric['recall'] = recall.mean()
            if 'precision' in evaluation_metric:
                evaluation_metric['precision'] = precision.mean()
            if 'f1_score' in evaluation_metric:
                evaluation_metric['f1_score'] = (2 * np.multiply(precision, recall) / (precision + recall + 1e-12)).mean()
        return evaluation_metric


# ------------------------------------------------------------------------


class SoftmaxCrossentropyLoss(Objective):
    _label = OBJECTIVE.SOFTMAX_CROSSENTROPY_LOSS
    """
    Objective using softmax (multinomial)crossentropyfor loss function.
    Arguments:
        size: objective size
        name: objective name
        metric: loss and accuracy metrics
    """
    @MType(size=int,
           name=str,
           metric=(str,))
    def __init__(self, *,
                 size=1,
                 name='',
                 metric=('loss', 'accuracy')):
        super().__init__(size=size, name=name)
        self.reconfig(metric=metric)

    # ------------------------------------------------------------------------

    @MType(shape=OneOfType((int,), None),
           metric=OneOfType((str,), None))
    def reconfig(self, *,
                 shape=None,
                 metric=None):
        """
        Reconfig objective
        Arguments:
            shape: objective layer shape
            metric: loss metric
        """
        if metric is not None:
            if 'loss' in metric or ('accuracy' or 'acc'):
                if 'loss' in metric:
                    self._evaluation['metric']['loss'] = 0
                if ('accuracy' or 'acc') in metric:
                    self._evaluation['metric']['accuracy'] = 0
                if ('recall' or 'rc') in metric:
                    self._evaluation['metric']['recall'] = 0
                if ('precision' or 'prec') in metric:
                    self._evaluation['metric']['precision'] = 0
                if ('f1_score' or 'f1') in metric:
                    self._evaluation['metric']['f1_score'] = 0
            else:
                raise TypeError('Unknown metric {metric} for objective {name}.'.format(metric=metric, name=self.name))
        if shape is not None:
            super().reconfig(shape=shape)
        self.reset()

    @MType(dict, np.ndarray, residue=dict)
    @MShape(axis=1)
    def forward(self, stage, a_t, *, residue={}):
        """
        Do forward pass method.
        Arguments:
            stage: forward stage
            a_t: post-nonlinearity (a) tensor
            residue:
        Returns:
            layer
        """
        exps_a_t = np.exp(a_t - a_t.max(axis=1, keepdims=True))
        softmax_a_t = exps_a_t / exps_a_t.sum(axis=1, keepdims=True)
        return super().forward(stage, softmax_a_t, residue=residue)

    @MType(np.ndarray, np.ndarray, dict)
    def compute_loss(self, y_t, y_prime_t, *, residue={}):
        """
        Compute the loss.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            residue:
        Returns:
            tuple
        """
        y_prime_t = y_prime_t.astype(np.float32)
        ly_t = -np.log(y_t[range(y_t.shape[0]), y_prime_t.argmax(axis=1)] + 1e-12)

        return (ly_t, residue)

    @MType(np.ndarray, np.ndarray, dict)
    def compute_loss_grad(self, y_t, y_prime_t, *, residue={}):
        """
        Compute the loss gradient tensor for gradient descent update.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            residue:
        Returns:
            tuple
        """
        ey_t = y_t - y_prime_t
        eyg_t = ey_t

        return (eyg_t, residue)

    @MType(np.ndarray, np.ndarray, np.ndarray, dict)
    def compute_evaluation_metric(self, y_t, y_prime_t, ly_t, evaluation_metric):
        """
        Compute the evaluation metric.
        Arguments:
            y_t: output (y) tensor
            y_prime_t: expected output (y) tensor
            ly_t: loss tensor
            evaluation_metric:
        Returns:
            metric
        """
        if 'loss' in evaluation_metric:
            evaluation_metric['loss'] += ly_t.mean()
        if 'accuracy' in evaluation_metric:
            evaluation_metric['accuracy'] += np.equal(y_prime_t.argmax(axis=1), y_t.argmax(axis=1)).astype(np.int8).mean()
        if 'recall' in evaluation_metric or 'precision' in evaluation_metric or 'f1_score' in evaluation_metric:
            y_t = np.round(y_t)
            true_pos = np.sum(np.multiply(y_t, y_prime_t), axis=0).astype(np.float)
            # true_neg = np.sum(np.multiply((1 - y_t), (1 - y_prime_t)), axis=0).astype(np.float)
            false_pos = np.sum(np.multiply(y_t, (1 - y_prime_t)), axis=0).astype(np.float)
            false_neg = np.sum(np.multiply((1 - y_t), y_prime_t), axis=0).astype(np.float)
            recall = true_pos / (true_pos + false_neg + 1e-12)
            precision = true_pos / (true_pos + false_pos + 1e-12)
            if 'recall' in evaluation_metric:
                evaluation_metric['recall'] = recall.mean()
            if 'precision' in evaluation_metric:
                evaluation_metric['precision'] = precision.mean()
            if 'f1_score' in evaluation_metric:
                evaluation_metric['f1_score'] = (2 * np.multiply(precision, recall) / (precision + recall + 1e-12)).mean()
        return evaluation_metric
