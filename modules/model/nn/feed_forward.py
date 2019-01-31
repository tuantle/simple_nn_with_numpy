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
import os
import warnings
import copy
import json
import time
import math
import numpy as np

from util.const import CONST
from util.validation import (
    MType,
    OneOfType
)

from npcore.layer.link import Link
from npcore.layer.gates import (
    Linear,
    Nonlinear
)
from npcore.layer.sockets import BatchNorm
from npcore.layer.objectives import (
    Objective,
    MSELoss,
    MAELoss,
    LogCoshLoss,
    XTanhLoss,
    AlgebraicLoss,
    SigmoidCrossentropyLoss,
    SoftmaxCrossentropyLoss
)
from npcore.optimizers import (
    OPTIMIZER,
    Optimizer
)
from npcore.regularizers import REGULARIZER
from model.sequencer import Sequencer

# ------------------------------------------------------------------------


class FEED_FORWARD(CONST):
    LABEL = 'feed_forward'

    DEFAULT_EPOCH_LIMIT = 50

# ------------------------------------------------------------------------


class FeedForward(type):
    """
    A metaclass for a base feed forward class.
    """
    def __getattr__(cls, key):
        return cls[key]

    def __setattr__(cls, key, value):
        raise ValueError('Cannot set value to a feed forward class.')

    @property
    def label(cls):
        """
        Get layer label.
        Returns:
            str
        """
        return cls._label


# ------------------------------------------------------------------------


class FeedForward(object, metaclass=FeedForward):
    _label = FEED_FORWARD.LABEL
    """
    A feed forward class.
    Arguments:
        name:
    """

    @MType(name=str)
    def __init__(self, *, name=''):
        self._name = name
        self._sequencer = None
        self._hparam = {
            'eta': OPTIMIZER.DEFAULT_ETA,
            'eta_decay': OPTIMIZER.DEFAULT_ETA_DECAY,
            'beta_decay1': OPTIMIZER.DEFAULT_BETA_DECAY1,
            'beta_decay2': OPTIMIZER.DEFAULT_BETA_DECAY2,
            'momentum': OPTIMIZER.DEFAULT_MOMENTUM,
            'l1_lambda':  REGULARIZER.DEFAULT_L1_LAMBDA,
            'l2_lambda': REGULARIZER.DEFAULT_L2_LAMBDA
        }
        self._setup_completed = False
        self._eta_scheduler = None
        self._monitor = None
        self._checkpoint = None

    def __str__(self):
        if self.name != '':
            return self.name + '_' + self.label
        else:
            return self.label

    # ------------------------------------------------------------------------

    @property
    def label(self):
        """
        Get feed forward label.
        Returns:
            str
        """
        return type(self).label

    @property
    def name(self):
        """
        Get feed forward name.
        Returns:
        """
        return self._name

    @name.setter
    @MType(str)
    def name(self, name):
        """
        Set feed forward name.
        Arguments:
            name: feed forward name
        """
        self._name = name

    @property
    def sequence(self):
        """
        Get feed forward sequence.
        Returns:
        """
        if self.is_valid:
            return self._sequencer.sequence
        else:
            return None

    @property
    def is_valid(self):
        """
        Check if feed forward has a valid sequence.
        Returns:
            bool
        """
        return self._sequencer is not None and self._sequencer.is_valid

    @property
    def is_complete(self):
        """
        Check if feed forward has a valid and complete sequence.
        Returns:
            bool
        """
        return self.is_valid and self._sequencer.is_complete and self._setup_completed

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return feed forward as a snapshot dict data.
        Arguments:
            as_json:
            beautify_json:
        Returns:
            dict
        """
        model_snapshot = {
            'name': self.name,
            'label': self.label,
            'base_label': FeedForward.label,
            'hparam': self._hparam,
            'sequencer': self._sequencer.snapshot(as_json=False, beautify_json=False) if self.is_complete else None
        }
        if as_json:
            if beautify_json:
                return json.dumps(model_snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(model_snapshot)
        else:
            return model_snapshot.copy()

    def unassign_hooks(self):
        """
        Unassign all callback functions.
        """
        self._eta_scheduler = None
        self._monitor = None
        self._checkpoint = None

    @MType(eta_scheduler=OneOfType(callable, None),
           checkpoint=OneOfType(callable, None),
           monitor=OneOfType(callable, None))
    def assign_hook(self, *,
                    eta_scheduler=None,
                    checkpoint=None,
                    monitor=None):
        """
        Assign callback functions.
        Arguments:
            eta_scheduler:
            checkpoint:
            monitor: callback function to retreive and monitor report/summary
        """
        if eta_scheduler is not None:
            self._eta_scheduler = eta_scheduler
        if checkpoint is not None:
            self._checkpoint = checkpoint
        if monitor is not None:
            self._monitor = monitor

    @MType(int)
    def on_epoch_begin(self, epoch):
        """
        Arguments:
            epoch:
        """
        pass

    @MType(int)
    def on_epoch_end(self, epoch):
        """
        Arguments:
            epoch:
        """
        pass

    def on_setup_completed(self):
        """
        """
        pass

    @abc.abstractmethod
    def construct(self):
        """
        """
        pass

    @property
    def summary(self):
        """
        Get feed forward summary.
        Returns:
        """
        nonlinear_gate_count = 0
        linear_gate_count = 0
        link_count = 0
        total_param_count = 0
        link_optim = ''
        layer_label = ''
        layer_shape = ''
        hdr = 'Layer\tIndex\tOptim\tType\t\tShape\t\tParams\n'
        div1 = '====================================================================================\n'
        div2 = '        ----------------------------------------------------------------------------\n'

        summary = '### Feed forward {name} Summary ###\n'.format(name=self.name)
        summary += hdr + div1

        if self.is_valid:
            if self.sequence.head.name != '':
                summary += '{name}:\n'.format(name=self.sequence.head.name)
            for layer in self.sequence.head:
                param_count = 0
                if isinstance(layer, Nonlinear):
                    nonlinear_gate_count += 1
                    param_count += layer.size
                if isinstance(layer, Linear):
                    linear_gate_count += 1
                if isinstance(layer, BatchNorm):
                    param_count += 2 * layer.shape[1]
                if isinstance(layer, Link):
                    link_count += 1
                    param_count += (layer.shape[0] * layer.shape[1]) + layer.shape[0]

                if isinstance(layer, Link):
                    link_optim = '{label}'.format(label=layer.optim.label)
                    if layer.is_frozen:
                        layer_label = '{label} (frozen)'.format(label=layer.label)
                    else:
                        layer_label = '{label}'.format(label=layer.label)
                elif isinstance(layer, Nonlinear):
                    link_optim = ''
                    layer_label = '{label}'.format(label=layer.label)
                else:
                    link_optim = ''
                    layer_label = '{label}'.format(label=layer.label)

                if isinstance(layer, Link):
                    layer_shape = str(layer.shape)
                else:
                    layer_shape = '(*, {size})'.format(size=layer.size)

                if layer.has_next:
                    summary += div2
                    summary += '\t{index:<8d}{label1:<8s}{label2:<16s}{shape:<16s}{param_count:<8d}\n'.format(
                        index=layer.index,
                        label1=link_optim,
                        label2=layer_label,
                        shape=layer_shape,
                        param_count=param_count)
                    if layer.next.name != '' and layer.next.name != layer.name and layer.next.has_next:
                        summary += '{name}:\n'.format(name=layer.next.name)
                else:
                    summary += div1
                total_param_count += param_count

        if self.is_complete:
            summary += 'Objective             : {label}\n'.format(label=self.sequence.tail.label)
        summary += 'Total number of params: {count}\n'.format(count=total_param_count)
        summary += 'Total number of layers: {count}\n'.format(count=linear_gate_count + nonlinear_gate_count + link_count)
        summary += '                        {count} nonlinear gate layers\n'.format(count=nonlinear_gate_count)
        summary += '                        {count} link layers\n'.format(count=link_count)
        return summary

    @MType(objective=OneOfType(str, Objective),
           metric=(str,),
           optim=OneOfType(str, Optimizer, None),
           hparam=OneOfType(dict, None))
    def setup(self, *,
              objective='mse',
              metric=('loss',),
              optim=None,
              hparam=None):
        """
        Setup the objective layer where the loss & loss gradient is calculated.
        Arguments:
            objective: objective layer
            metric:
            optim:
            hparam:
        Returns:
            self
        """
        if self.is_complete:
            warnings.warn(
                'Feed forward {name} sequence is completed and already setup. Setup skipped.'.format(name=self.name),
                UserWarning)
        else:
            if not self.is_valid:
                sequencer = self.construct()
                if not sequencer.is_valid:
                    raise RuntimeError('Constructed sequence from sequencer {name} is invalid.'.format(name=sequencer.name))
                self._sequencer = sequencer

            if 'linear' != self.sequence.tail.label:
                warnings.warn(
                    'Output sequence of sequencer {name} is not linear.'.format(name=sequencer.name),
                    UserWarning)

            size = self.sequence.tail.size
            if isinstance(objective, str):
                name = self._sequencer.name + '_' + Objective.label
                objective_label = objective
                if MAELoss.label == objective_label:
                    self.sequence.tail.connect(MAELoss(size=size,
                                                       name=name,
                                                       metric=metric)).lock()
                elif MSELoss.label == objective_label:
                    self.sequence.tail.connect(MSELoss(size=size,
                                                       name=name,
                                                       metric=metric)).lock()
                elif LogCoshLoss.label == objective_label:
                    self.sequence.tail.connect(LogCoshLoss(size=size,
                                                           name=name,
                                                           metric=metric)).lock()
                elif XTanhLoss.label == objective_label:
                    self.sequence.tail.connect(XTanhLoss(size=size,
                                                         name=name,
                                                         metric=metric)).lock()
                elif AlgebraicLoss.label == objective_label:
                    self.sequence.tail.connect(AlgebraicLoss(size=size,
                                                             name=name,
                                                             metric=metric)).lock()
                elif SigmoidCrossentropyLoss.label == objective_label:
                    self.sequence.tail.connect(SigmoidCrossentropyLoss(size=size,
                                                                       name=name,
                                                                       metric=metric)).lock()
                elif SoftmaxCrossentropyLoss.label == objective_label:
                    self.sequence.tail.connect(SoftmaxCrossentropyLoss(size=size,
                                                                       name=name,
                                                                       metric=metric)).lock()
                else:
                    raise TypeError('Unknown objective {objective_label} for objective layer.'.format(objective_label=objective_label))
            else:
                if size != objective.size:
                    objective.reconfig(shape=(1, size))

                if metric is not None and metric != tuple(objective.evaluation_metric.keys()):
                    objective.reconfig(metric=metric)
                    warnings.warn(
                        'Overiding custom objective layer {name} metric. Using metric {metric}.'.format(name=objective.name, metric=metric),
                        UserWarning)

                self.sequence.tail.connect(objective).lock()
                self.sequence.tail.name = self._sequencer.name + objective.name

            self._setup_completed = True

            self.reconfig(optim=optim, hparam=hparam)
            self.on_setup_completed()
        return self

    @MType(optim=OneOfType(str, Optimizer, None), hparam=OneOfType(dict, None))
    def reconfig(self, *,
                 optim=None,
                 hparam=None):
        """
        Arguments:
            optim:
            hparam:
        Returns:
            self
        """
        if not self.is_complete:
            raise RuntimeError('Feed forward {name} sequence is incomplete. Need to complete setup.'.format(name=self.name))

        if hparam is not None:
            if 'eta' in hparam:
                if hparam['eta'] <= 0:
                    warnings.warn(
                        'Learning rate eta cannot be <= 0. Reset to {eta}.'.format(eta=OPTIMIZER.DEFAULT_ETA),
                        UserWarning)
                    hparam['eta'] = OPTIMIZER.DEFAULT_ETA
            if 'eta_decay' in hparam:
                if hparam['eta_decay'] < 0:
                    warnings.warn(
                        'Learning rate eta decay cannot be < 0. Reset to {eta_decay}.'.format(eta_decay=OPTIMIZER.DEFAULT_ETA_DECAY),
                        UserWarning)
                    hparam['eta_decay'] = OPTIMIZER.DEFAULT_ETA_DECAY
            if 'beta_decay1' in hparam:
                if hparam['beta_decay1'] < 0:
                    warnings.warn(
                        'Optimization beta decay cannot be < 0. Reset to {beta_decay}.'.format(beta_decay=OPTIMIZER.DEFAULT_BETA_DECAY1),
                        UserWarning)
                    hparam['beta_decay1'] = OPTIMIZER.DEFAULT_BETA_DECAY1
            if 'beta_decay2' in hparam:
                if hparam['beta_decay2'] < 0:
                    warnings.warn(
                        'Optimization beta decay cannot be < 0. Reset to {beta_decay}.'.format(beta_decay=OPTIMIZER.DEFAULT_BETA_DECAY2),
                        UserWarning)
                    hparam['beta_decay2'] = OPTIMIZER.DEFAULT_BETA_DECAY1
            if 'momentum' in hparam:
                if hparam['momentum'] < 0:
                    warnings.warn(
                        'Optimization momentum cannot be < 0. Reset to {momentum}.'.format(momentum=OPTIMIZER.DEFAULT_MOMENTUM),
                        UserWarning)
                    hparam['momentum'] = OPTIMIZER.DEFAULT_MOMENTUM
            if 'l1_lambda' in hparam:
                if hparam['l1_lambda'] < 0:
                    warnings.warn(
                        'Regularization lambda cannot be < 0. Reset to {l_lambda}.'.format(l_lambda=OPTIMIZER.DEFAULT_L1_LAMBDA),
                        UserWarning)
                    hparam['l1_lambda'] = OPTIMIZER.DEFAULT_L1_LAMBDA
            if 'l2_lambda' in hparam:
                if hparam['l2_lambda'] < 0:
                    warnings.warn(
                        'Regularization lambda cannot be < 0. Reset to {l_lambda}.'.format(l_lambda=OPTIMIZER.DEFAULT_L2_LAMBDA),
                        UserWarning)
                    hparam['l2_lambda'] = OPTIMIZER.DEFAULT_L2_LAMBDA
            self._hparam.update(hparam)

        if optim is not None:
            self._sequencer.reconfig_all(optim=optim)

    @MType(int, int)
    def compute_eta(self, epoch, epoch_limit):
        """
        Get current learning rate
        Arguments:
            epoch:
            epoch_limit:
        Returns:
            eta: learning rate
        """
        eta = self._hparam['eta']
        if self._eta_scheduler is not None:
            eta = self._eta_scheduler(epoch, epoch_limit, eta)
            if not isinstance(eta, float) or eta < 0:
                raise TypeError('Learning rate value must be a positive floating point number.')
        else:
            eta_decay = self._hparam['eta_decay']
            if eta_decay > 0:
                eta *= math.pow(eta_decay, epoch / epoch_limit)
        return eta

    @MType(np.ndarray)
    def predict(self, x_t):
        """
        Do forward prediction with a given input tensor.
        Arguments:
            x_t: input tensor
        Returns:
            y_t: output prediction tensor
        """
        if not self.is_complete:
            raise RuntimeError('Feed forward {name} sequence is incomplete. Need to complete setup.'.format(name=self.name))
        if len(x_t.shape) != 2:
            raise RuntimeError('Input tensor shape size is invalid. Input tensor shape must have a length of 2.')

        (input_sample_size, input_feature_size) = x_t.shape
        if input_feature_size != self.sequence.head.size:
            raise ValueError('Input tensor feature size does not match the size of input layer of {size}.'.format(size=self.sequence.head.size))
        # x_t = x_t.copy()
        stage = {
            'epoch': 0,
            'mode': 'predicting',
            'hparam': copy.deepcopy(self._hparam)
        }
        # tstart_ns = time.process_time_ns()
        tstart_us = time.process_time()
        self.sequence.head.forward(stage, x_t)
        # tend_ns = time.process_time_ns()
        tend_us = time.process_time()
        # elapse_per_epoch_ms = int(round((tend_ns - tstart_ns) * 0.000001))
        elapse_per_epoch_ms = int(round((tend_us - tstart_us) * 1000))

        if self._monitor is not None:
            report = {
                'name': self.name,
                'stage': stage,
                'elapse': {
                    'per_epoch_ms': elapse_per_epoch_ms,
                    'total_ms': elapse_per_epoch_ms,
                }
            }
            self._monitor(report)

        return self.sequence.tail.outputs

    @MType(np.ndarray, np.ndarray,
           epoch_limit=int,
           batch_size=int,
           tl_split=float, tl_shuffle=bool,
           verbose=bool)
    def learn(self, x_t, y_prime_t, *,
              epoch_limit=FEED_FORWARD.DEFAULT_EPOCH_LIMIT,
              batch_size=1,
              tl_split=0, tl_shuffle=False,
              verbose=True):
        """
        Arguments:
            x_t:
            y_prime_t:
            epoch_limit:
            batch_size:
            tl_split:
            tl_shuffle:
            verbose:
        """
        if not self.is_complete:
            raise RuntimeError('Feed forward {name} sequence is incomplete. Need to complete setup.'.format(name=self.name))
        if len(x_t.shape) != 2:
            raise RuntimeError('Input tensor shape size is invalid. Input tensor shape must have a length of 2.')
        elif len(y_prime_t.shape) != 2:
            raise RuntimeError('Expected output tensor shape size is invalid. Output tensor shape must have a length of 2.')

        (input_sample_size, input_feature_size) = x_t.shape
        (expected_output_sample_size, expected_output_prediction_size) = y_prime_t.shape
        if input_feature_size != self.sequence.head.size:
            raise ValueError('Input tensor feature size does not match the size of input layer of {size}.'.format(size=self.sequence.head.size))
        if expected_output_prediction_size != self.sequence.tail.size:
            raise ValueError('Expected output tensor prediction size does not match the size of output layer of {size}.'.format(size=self.sequence.tail.size))
        if expected_output_sample_size != input_sample_size:
            raise ValueError('Input and output tensor sample sizes do not matched.')
        if tl_shuffle:
            shuffler = np.random.permutation(input_sample_size)
            x_t = x_t[shuffler]  # .copy()
            y_prime_t = y_prime_t[shuffler]  # .copy()
        # else:
        #     x_t = x_t.copy()
        #     y_prime_t = y_prime_t.copy()

        if tl_split < 0 or tl_split > 0.5:
            tl_split = 0
            warnings.warn(
                'Testing and learning split ratio must be >= 0 and <= 0.5. Reset testing and learning split ratio to 0.',
                UserWarning)

        enable_testing = tl_split > 0

        if enable_testing:
            if input_sample_size == 1:
                learning_sample_size = input_sample_size
                enable_testing = False
                warnings.warn(
                    'Input sample size = 1. Reset testing and learning split ratio to 0.',
                    UserWarning)
            else:
                learning_sample_size = int(input_sample_size * (1 - tl_split))
                learning_sample_size = learning_sample_size - learning_sample_size % batch_size
                testing_sample_size = input_sample_size - learning_sample_size
        else:
            learning_sample_size = input_sample_size

        if batch_size < 1 or batch_size > learning_sample_size:
            batch_size = learning_sample_size
            warnings.warn(
                'Batch size must be >= 1 and <= learning sample size {size}. Set batch size = learning sample size.'.format(size=learning_sample_size),
                UserWarning)

        stop_learning = False
        stage = {
            'epoch': 0,
            'mode': '',
            'hparam': copy.deepcopy(self._hparam)
        }
        stage['hparam']['batch_size'] = batch_size
        elapse_total_ms = 0

        for layer in self.sequence.head:
            if isinstance(layer, Link) or isinstance(layer, BatchNorm):
                layer.optim.reset()

        for epoch in range(epoch_limit):

            self.on_epoch_begin(epoch)

            tstart_us = time.process_time()
            # tstart_ns = time.process_time_ns()

            self.sequence.tail.reset()

            stage['epoch'] = epoch
            stage['mode'] = 'learning'
            stage['hparam']['eta'] = self.compute_eta(epoch, epoch_limit)

            if batch_size == learning_sample_size:
                batched_x_t = x_t[:learning_sample_size]
                batched_y_prime_t = y_prime_t[:learning_sample_size]
                self.sequence.head.forward(stage, batched_x_t).evaluate(batched_y_prime_t).backward(stage)
            else:
                for i in range(learning_sample_size):
                    if (i + batch_size) < learning_sample_size:
                        batched_x_t = x_t[i: i + batch_size]
                        batched_y_prime_t = y_prime_t[i: i + batch_size]
                    # else:
                    #     batched_x_t = x_t[i: learning_sample_size]
                    #     batched_y_prime_t = y_prime_t[i: learning_sample_size]

                    self.sequence.head.forward(stage, batched_x_t).evaluate(batched_y_prime_t).backward(stage)

            learning_evaluation_metric = self.sequence.tail.evaluation_metric

            if enable_testing:
                stage['mode'] = 'learning_and_testing'
                self.sequence.tail.reset()
                self.sequence.head.forward(stage, x_t[learning_sample_size:]).evaluate(y_prime_t[learning_sample_size:])

                testing_evaluation_metric = self.sequence.tail.evaluation_metric

                if self._checkpoint is not None:
                    stop_learning = self._checkpoint(epoch, learning_evaluation_metric, testing_evaluation_metric)
                else:
                    stop_learning = False
            else:
                if self._checkpoint is not None:
                    stop_learning = self._checkpoint(epoch, learning_evaluation_metric, None)
                else:
                    stop_learning = False

            tend_us = time.process_time()
            elapse_per_epoch_ms = int(round((tend_us - tstart_us) * 1000))
            # tend_ns = time.process_time_ns()
            # elapse_per_epoch_ms = int(round((tend_ns - tstart_ns) * 0.000001))
            elapse_total_ms += elapse_per_epoch_ms

            self.on_epoch_end(epoch)

            if self._monitor is not None:
                stage['mode'] = 'learning'
                self.sequence.tail.reset()
                self.sequence.head.forward(stage, x_t[:learning_sample_size])
                snapshot_learning_output_t = self.sequence.tail.outputs
                report = {
                    'name': self.name,
                    'stage': stage,
                    'epoch_limit': epoch_limit,
                    'learning_sample_size': learning_sample_size,
                    'snapshot': {
                        'learning': {
                            'inputs': x_t[:learning_sample_size],
                            'expected_outputs': y_prime_t[:learning_sample_size],
                            'outputs': snapshot_learning_output_t
                        }
                    },
                    'elapse': {
                        'per_epoch_ms': elapse_per_epoch_ms,
                        'total_ms': elapse_total_ms,
                    },
                    'evaluation_metric': {
                        'learning': learning_evaluation_metric
                    }
                }
                if enable_testing:
                    stage['mode'] = 'learning_and_testing'
                    self.sequence.tail.reset()
                    self.sequence.head.forward(stage, x_t[learning_sample_size:])
                    snapshot_testing_output_t = self.sequence.tail.outputs
                    report['test_sample_size'] = testing_sample_size
                    report['snapshot']['testing'] = {
                        'inputs': x_t[learning_sample_size:],
                        'expected_outputs': y_prime_t[learning_sample_size:],
                        'outputs': snapshot_testing_output_t
                    }
                    report['evaluation_metric']['testing'] = testing_evaluation_metric
                self._monitor(report)

            if verbose:
                print('Epoch: {epoch}/{epoch_limit} - Elapse/Epoch: {elapse_per_epoch} ms - Elapse: {elapse_total} s'.format(epoch=epoch + 1,
                                                                                                                             epoch_limit=epoch_limit,
                                                                                                                             elapse_per_epoch=elapse_per_epoch_ms,
                                                                                                                             elapse_total=round(elapse_total_ms * 1e-3)), end='\n', flush=True)
                print('\tLearning rate: {eta:.9f}'.format(eta=stage['hparam']['eta']), end='\n', flush=True)
                if enable_testing:
                    learning_metric_summary = ''
                    testing_metric_summary = ''
                    for (metric_name, metric_value) in learning_evaluation_metric.items():
                        learning_metric_summary += '{metric_name}: {metric_value:.9f} '.format(metric_name=metric_name,
                                                                                               metric_value=metric_value)
                    for (metric_name, metric_value) in testing_evaluation_metric.items():
                        testing_metric_summary += '{metric_name}: {metric_value:.9f} '.format(metric_name=metric_name,
                                                                                              metric_value=metric_value)
                    print('\tLearning {metric_summary}'.format(metric_summary=learning_metric_summary), end='\n', flush=True)
                    print('\tTesting {metric_summary}'.format(metric_summary=testing_metric_summary), end='\n', flush=True)
                else:
                    learning_metric_summary = ''
                    for (metric_name, metric_value) in learning_evaluation_metric.items():
                        learning_metric_summary += '{metric_name}: {metric_value:.9f} '.format(metric_name=metric_name,
                                                                                               metric_value=metric_value)
                    print('\tLearning {metric_summary}'.format(metric_summary=learning_metric_summary), end='\n', flush=True)
                if epoch == epoch_limit - 1:
                    print('\n')
            if stop_learning:
                break

    @MType(str, save_as=OneOfType(str, None))
    def save_snapshot(self, filepath, *, save_as=None):
        """
        Save model snapshot to file.
        Arguments:
            filepath:
            save_as:
        """
        if not self.is_complete:
            raise RuntimeError('Feed forward {name} sequence is incomplete. Need to complete setup.'.format(name=self.name))
        if save_as is not None and save_as != '':
            filename = os.path.join(filepath, save_as + '.json')
        else:
            if self.name != '':
                filename = os.path.join(filepath, self.name + '.json')
            else:
                filename = os.path.join(filepath, 'untitled.json')

        with open(filename, 'w') as file:
            model_snapshot = self.snapshot(as_json=False, beautify_json=True)
            json.dump(model_snapshot, file, ensure_ascii=False)

    @MType(str, overwrite=bool)
    def load_snapshot(self, filename, *, overwrite=False):
        """
        Load model snapshot from file.
        Arguments:
            filename:
            overwrite:
        Returns:
            self
        """
        if self.is_valid and not overwrite:
            raise RuntimeError('Feed forward {name} sequence is valid. Cannot overwrite sequence.'.format(name=self.name))
        with open(filename, 'r') as file:
            model_snapshot = json.load(file)
            hparam = model_snapshot['hparam']
            sequencer_snapshot = model_snapshot['sequencer']
            self._setup_completed = False
            self._sequencer = Sequencer().load_snapshot(sequencer_snapshot, overwrite=overwrite)

            sequence_snapshot = sequencer_snapshot['sequences'][-1]
            objective_label = sequence_snapshot['base_label']
            if Objective.label in objective_label:
                objective = sequence_snapshot['label']
                metric = tuple(sequence_snapshot['metric'])
                self.setup(objective=objective,
                           metric=metric,
                           hparam=hparam)
            self.name = model_snapshot['name']

        return self
