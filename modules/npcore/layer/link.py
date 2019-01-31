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

import json
import warnings
import numpy as np

from util.const import CONST
from npcore.layer.layer import Layer

from util.validation import (
    MShape,
    MType,
    OneOfType
)
from npcore.initializers import (
    Initializer,
    Zeros, Ones, Constant, Identity,
    RandomNormal, RandomUniform,
    GlorotRandomNormal, GlorotRandomUniform
)
from npcore.optimizers import (
    Optimizer,
    SGD, SGDM,
    RMSprop, Adam
)
from npcore.regularizers import (
    Regularizer,
    L1Lasso,
    L2Ridge,
    L1L2ElasticNet
)

# ------------------------------------------------------------------------


class LINK(CONST):
    LABEL = 'link'
    ARRANGEMENT = ('1', '0')


# ------------------------------------------------------------------------


class Link(Layer):
    _label = LINK.LABEL
    _arrangement = LINK.ARRANGEMENT
    """
    A fully connected link that connect two layers together consited of a weight matrix and bias vector.
    Arguments:
        shape:
        name:
        weight_init: weight matrix initializer
        weight_reg: weight matrix regularization
        bias_init: bias vector initializer
        optim:
    """
    @MType(shape=(int,),
           name=str,
           weight_init=OneOfType(str, Initializer),
           weight_reg=OneOfType(str, Regularizer, None),
           bias_init=OneOfType(str, float, Initializer),
           optim=OneOfType(str, Optimizer))
    def __init__(self, *,
                 shape=(1, 1),
                 name='',
                 weight_init='random_normal',
                 weight_reg='not_use',
                 bias_init='zeros',
                 optim='sgd'):
        self._frozen = False
        self._weight_init = None
        self._weight_reg = None
        self._bias_init = None
        self._w_m = None
        self._b_v = None
        self._optim = None

        self._monitor = None

        super().__init__(shape=shape, name=name)
        self.reconfig(weight_init=weight_init,
                      weight_reg=weight_reg,
                      bias_init=bias_init,
                      optim=optim)

    def __str__(self):
        return super().__str__() + '_' + LINK.LABEL

    # ------------------------------------------------------------------------

    @property
    def is_frozen(self):
        """
        Check if layer is frozen.
        Returns:
            is frozen flag
        """
        return self._frozen

    def freeze(self):
        """
        Freeze layer
        """
        self._frozen = True

    def unfreeze(self):
        """
        Unfreeze layer
        """
        self._frozen = False

    @property
    def inputs(self):
        """
        Get link forward pass input tensor.
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
        Get link forward pass output tensor.
        Returns:
            tensor
        """
        if self.has_next:
            return self.next.inputs
        else:
            return None

    @property
    def weights(self):
        """
        Get link weight matrix.
        Returns:
            matrix
        """
        if self._w_m is not None:
            return self._w_m.copy()
        else:
            return None

    @weights.setter
    @MType(np.ndarray)
    @MShape(axis=-1)
    def weights(self, w_m):
        """
        Set link weight matrix.
        """
        if self.is_frozen:
            warnings.warn(
                'Cannot set weights to a frozen link {name}.'.format(name=self.name),
                UserWarning)
        else:
            np.copyto(self._w_m, w_m, casting='same_kind')

    @property
    def biases(self):
        """
        Get link bias vector.
        Returns:
            vector
        """
        if self._b_v is not None:
            return self._b_v.copy()
        else:
            return None

    @biases.setter
    @MType(np.ndarray)
    @MShape(axis=1)
    def biases(self, b_v):
        """
        Set link bias vector.
        """
        if self.is_frozen:
            warnings.warn(
                'Cannot set biases to a frozen link {name}.'.format(name=self.name),
                UserWarning)
        else:
            np.copyto(self._b_v, b_v, casting='same_kind')

    @property
    def optim(self):
        """
        Get link optimizer.
        Returns:
            optimizer
        """
        return self._optim

    def reset(self):
        """
        Reset internal evaluation states.
        """
        if self._weight_init is not None:
            self._w_m = self._weight_init(self.shape)
        if self._bias_init is not None:
            self._b_v = self._bias_init((1, self.size))
        if self._optim is not None:
            self._optim.reset()

    @MType(shape=OneOfType((int,), None),
           weight_init=OneOfType(str, Initializer, None),
           weight_reg=OneOfType(str, Regularizer, None),
           bias_init=OneOfType(str, float, Initializer, None),
           optim=OneOfType(str, Optimizer, None))
    def reconfig(self, *,
                 shape=None,
                 weight_init=None,
                 weight_reg=None,
                 bias_init=None,
                 optim=None):
        """
        Reconfig link
        Arguments:
            shape
            ;
            weight_init:
            weight_reg:
            bias_init:
            optim:
        """
        if self.is_frozen:
            warnings.warn(
                'Link {name} is frozen. Reconfig link skipped.'.format(name=self.name),
                UserWarning)
        else:
            if weight_init is not None:
                if isinstance(weight_init, str):
                    weight_init_label = weight_init
                    if self._weight_init is not None and weight_init_label == self._weight_init.label:
                        warnings.warn(
                            'No change made to link weight initializer. Re-initializing link weights skipped.',
                            UserWarning)
                    else:
                        if Zeros.label == weight_init_label:
                            self._weight_init = Zeros()
                        elif Ones.label == weight_init_label:
                            self._weight_init = Ones()
                        elif Identity.label == weight_init_label:
                            self._weight_init = Identity()
                        elif RandomNormal.label == weight_init_label:
                            self._weight_init = RandomNormal()
                        elif RandomUniform.label == weight_init_label:
                            self._weight_init = RandomUniform()
                        elif GlorotRandomNormal.label == weight_init_label:
                            self._weight_init = GlorotRandomNormal()
                        elif GlorotRandomUniform.label == weight_init_label:
                            self._weight_init = GlorotRandomUniform()
                        else:
                            raise TypeError('Unknown weight initializer {weight_init_label} for link {name}.'.format(weight_init_label=weight_init_label, name=self.name))
                        self._w_m = self._weight_init(self.shape)
                else:
                    if self._weight_init is not None and weight_init.label == self._weight_init.label:
                        warnings.warn(
                            'No change made to link weight initializer. Re-initializing link weights skipped.',
                            UserWarning)
                    else:
                        self._weight_init = weight_init
                        self._w_m = self._weight_init(self.shape)
            if weight_reg is not None:
                if isinstance(weight_reg, str):
                    weight_reg_label = weight_reg
                    if self._weight_reg is not None and weight_reg_label == self._weight_reg.label:
                        warnings.warn(
                            'No change made to link weight regularizer. Reconfig link weight regularizer skipped.',
                            UserWarning)
                    else:
                        if weight_reg_label == 'not_use':
                            self._weight_reg = None
                        elif L1Lasso.label == weight_reg_label:
                            self._weight_reg = L1Lasso()
                        elif L2Ridge.label == weight_reg_label:
                            self._weight_reg = L2Ridge()
                        elif L1L2ElasticNet.label == weight_reg_label:
                            self._weight_reg = L1L2ElasticNet()
                        else:
                            raise TypeError('Unknown weight regularizer {weight_reg_label} for link {name}.'.format(weight_reg_label=weight_reg_label, name=self.name))
                else:
                    if self._weight_reg is not None and weight_reg.label == self._weight_reg.label:
                        warnings.warn(
                            'No change made to link weight initializer. Reconfig link weight regularizer skipped.',
                            UserWarning)
                    else:
                        self._weight_reg = weight_reg
            if bias_init is not None:
                if isinstance(bias_init, str):
                    bias_init_label = bias_init
                    if self._bias_init is not None and bias_init_label == self._bias_init.label:
                        warnings.warn(
                            'No change made to link bias initializer. Re-initializing link biases skipped.',
                            UserWarning)
                    else:
                        if bias_init_label == 'not_use':
                            self._bias_init = None
                        elif Zeros.label == bias_init_label:
                            self._bias_init = Zeros()
                        elif Ones.label == bias_init_label:
                            self._bias_init = Ones()
                        else:
                            raise TypeError('Unknown bias initializer {bias_init_label} for link {name}.'.format(bias_init_label=bias_init_label, name=self.name))
                        if self._bias_init is not None:
                            self._b_v = self._bias_init((1, self.size))
                        else:
                            self._b_v = None
                elif isinstance(bias_init, float):
                    self._bias_init = Constant(bias_init)
                    self._b_v = self._bias_init((1, self.size))
                else:
                    if self._bias_init is not None and bias_init.label == self._bias_init.label:
                        warnings.warn(
                            'No change made to link bias initializer. Re-initializing link biases skipped.',
                            UserWarning)
                    else:
                        self._bias_init = bias_init
                        self._b_v = self._bias_init((1, self.size))
            if optim is not None:
                if isinstance(optim, str):
                    optim_label = optim
                    if self._optim is not None and optim_label == self._optim.label:
                        warnings.warn(
                            'No change made to link optimizer. Reconfig link optimization skipped.',
                            UserWarning)
                    else:
                        if SGD.label == optim_label:
                            self._optim = SGD()
                        elif SGDM.label == optim_label:
                            self._optim = SGDM()
                        elif RMSprop.label == optim_label:
                            self._optim = RMSprop()
                        elif Adam.label == optim_label:
                            self._optim = Adam()
                        else:
                            raise TypeError('Unknown optimizer {optim_label} for link {name}.'.format(optim_label=optim_label, name=self.name))
                else:
                    if self._optim is not None and optim.label == self._optim.label:
                        warnings.warn(
                            'No change made to link optimizer. Reconfig link optimization skipped.',
                            UserWarning)
                    else:
                        self._optim = optim
            if shape is not None:
                super().reconfig(shape=shape)
            self.reset()

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return link as a snapshot dict data.
        Arguments:
            as_json:
            beautify_json:
        Returns:
            dict
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'base_label': Link.label + '_' + snapshot['base_label'],
            'frozen': self.is_frozen,
            'weight': {
                'dtype': str(self.weights.dtype),
                'values': self.weights.tolist()
            } if self.weights is not None else None,
            'bias': {
                'dtype': str(self.biases.dtype),
                'values': self.biases.tolist()
            } if self.biases is not None else None,
            'weight_init': self._weight_init.snapshot(as_json=False, beautify_json=False),
            'weight_reg': self._weight_reg.snapshot(as_json=False, beautify_json=False) if self._weight_reg is not None else None,
            'bias_init': self._bias_init.snapshot(as_json=False, beautify_json=False) if self._bias_init is not None else None,
            'optim': self._optim.snapshot(as_json=False, beautify_json=False)
        })

        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()

    def unassign_hooks(self):
        """
        Unassign all callback functions.
        """
        self._monitor = None

    @MType(monitor=OneOfType(callable, None))
    def assign_hook(self, *,
                    monitor=None):
        """
        Assign callback functions.
        Arguments:
            monitor: callback function to do probing during forward/backward pass
        """
        if monitor is not None:
            self._monitor = monitor

    @MType(dict, np.ndarray, residue=dict)
    @MShape(axis=1, transpose=True)
    def forward(self, stage, a_t, *, residue={}):
        """
        Do forward forward pass by calculating the weight sum of the pre-nonlinearity (z) tensor.
        Arguments:
            stage: forward stage
            a_t: post-nonlinearity (a) tensor
            residue:
        Returns:
            layer
        """
        if self.has_next:
            if self._bias_init is not None:
                z_t = np.inner(a_t, self._w_m.transpose()) + self._b_v
            else:
                z_t = np.inner(a_t, self._w_m.transpose())

            if self._monitor is not None:
                report = {
                    'pass': 'forward',
                    'stage': stage,
                    'inputs': self.inputs,
                    'outputs': self.outputs,
                    'weights': self.weights,
                    'biases': self.biases,
                    'residue': residue
                }
                self._monitor(report)

            return self.next.forward(stage, z_t, residue=residue)
        else:
            warnings.warn(
                'Dense link {name} connection is incomplete. Missing connection to next layer.'.format(name=self.name),
                UserWarning)
            return self

    @MType(dict, np.ndarray, np.ndarray, residue=dict)
    @MShape(axis=1, transpose=False)
    def backward(self, stage, azg_t, eag_t, *, residue={}):
        """
        Do backward backward pass by calculate error gradient tensor w.r.t. nonlinearity.
        Arguments:
            stage: backward stage
            azg_t: gradient post-nonlinearity (a) tensor w.r.t. pre-nonlinearity (z) tensor
            eag_t: gradient error tensor w.r.t. post-nonlinearity (a) tensor
            residue:
        Returns:
            layer
        """
        if self.has_prev:
            delta_t = np.multiply(eag_t, azg_t)
            if not self.is_frozen:
                if 'epoch' in stage:
                    epoch = stage['epoch']
                else:
                    raise ValueError('Input stage is missing the required epoch number.')

                hparam = stage['hparam']
                batch_size = hparam['batch_size']
                zwg_t = self.inputs

                if self._bias_init is not None:
                    if batch_size == 1:
                        ewg_m = np.multiply(zwg_t.transpose(), delta_t)
                        ebg_v = delta_t
                    else:
                        ewg_m = np.inner(zwg_t.transpose(), delta_t.transpose())
                        ebg_v = delta_t.mean(axis=0)

                    [w_delta_m, b_delta_v] = self._optim.compute_grad_descent_step(epoch, [ewg_m, ebg_v], hparam)

                    if self._weight_reg is not None:
                        w_reg_m = self._weight_reg.compute_regularization(epoch, self._w_m, hparam)
                        self._w_m -= w_delta_m + w_reg_m
                    else:
                        self._w_m -= w_delta_m

                    self._b_v -= b_delta_v
                else:
                    if batch_size == 1:
                        ewg_m = np.multiply(zwg_t.transpose(), delta_t)
                    else:
                        ewg_m = np.inner(zwg_t.transpose(), delta_t.transpose())
                    [w_delta_m] = self._optim.compute_grad_descent_step(epoch, [ewg_m], hparam)

                    if self._weight_reg is not None:
                        w_reg_m = self._weight_reg.compute_regularization(epoch, self._w_m, hparam)
                        self._w_m -= w_delta_m + w_reg_m
                    else:
                        self._w_m -= w_delta_m

            eag_t = np.inner(self._w_m, delta_t).transpose()

            if self._monitor is not None:
                report = {
                    'pass': 'backward',
                    'stage': stage,
                    'weights': self.weights,
                    'biases': self.biases,
                    'grad': {
                        'delta': delta_t,
                        'az': azg_t,
                        'ea': eag_t
                    },
                    'residue': residue
                }
                self._monitor(report)

            return self.prev.backward(stage, eag_t, residue=residue)
        else:
            warnings.warn(
                'Dense link {name} connection is incomplete. Missing connection to previous layer.'.format(name=self.name),
                UserWarning)
            return self
