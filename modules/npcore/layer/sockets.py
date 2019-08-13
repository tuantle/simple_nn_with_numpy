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
from npcore.initializers import Initializer, Zeros, Ones, Constant
from npcore.initializers import RandomBinary, RandomNormal, RandomUniform, GlorotRandomNormal, GlorotRandomUniform
from npcore.optimizers import Optimizer, SGD, SGDM, RMSprop, Adam

# ------------------------------------------------------------------------


class SOCKET(CONST):
    LABEL = 'socket'
    DROPOUT_LABEL = 'dropout'
    BATCH_NORMALIZER_LABEL = 'batch_norm'

    ARRANGEMENT = ('2', '0:1:2')

    DEFAULT_DROPOUT_PZERO = 0.5
    DEFAULT_BATCH_NORMALIZER_MOVING_MOMENTUM = 0.99

# ------------------------------------------------------------------------


class Socket(Layer):
    _label = SOCKET.LABEL
    _arrangement = SOCKET.ARRANGEMENT
    """
    Abtraction of a base socket layer.
    Arguments:
        shape: socket shape
        name: socket name
    """
    @MType(shape=(int,),
           name=str)
    def __init__(self, *,
                 shape=(1, 1),
                 name=''):
        self._a_t = None
        self._monitor = None
        super().__init__(shape=shape, name=name)

    def __str__(self):
        return super().__str__() + '_' + SOCKET.LABEL

    # ------------------------------------------------------------------------

    @property
    def inputs(self):
        """
        Get socket forward pass input tensor
        Returns:
        """
        if self.has_prev:
            return self.prev.outputs
        else:
            return None

    @property
    def outputs(self):
        """
        Get socket forward pass output tensor
        """
        if self._a_t is not None:
            return self._a_t.copy()
        else:
            return None

    def reset(self):
        """
        Reset internal evaluation states
        """
        self._a_t = None

    def unassign_hooks(self):
        """
        Unassign all callback functions
        """
        self._monitor = None

    @MType(monitor=OneOfType(callable, None))
    def assign_hook(self, *, monitor=None):
        """
        Assign callback functions
        Arguments:
            monitor: callback function to do probing during forward/backward pass
        """
        if monitor is not None:
            self._monitor = monitor

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return socket as a snapshot dict data
        Arguments:
            as_json:
            beautify_json:
        Returns:
            snapshot
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'base_label': Socket.label + '_' + snapshot['base_label']
        })
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()

    @MType(dict, np.ndarray, residue=dict)
    @MShape(axis=1, transpose=False)
    def forward(self, stage, a_t, *, residue={}):
        """
        Do forward pass by passing through the input (a) tensor
        Arguments:
            stage: forward stage
            a_t: post-nonlinearity (a) tensor
            residue:
        Returns:
            tail
        """
        (a_t, residue) = self.compute_forward_ops(stage, a_t, residue=residue)
        self._a_t = a_t

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
            warnings.warn(f'Socket {self.name} connection is incomplete. Missing connection to next layer.', UserWarning)
            return self

    @MType(dict, np.ndarray, residue=dict)
    @MShape(axis=1, transpose=False)
    def backward(self, stage, eag_t, *, residue={}):
        """
        Do backward backward pass by passing through the error gradient tensor w.r.t. nonlinearity
        Arguments:
            stage: backward stage
            eag_t: gradient error tensor w.r.t. post-nonlinearity (a) tensor
            residue:
        Returns:
            head
        """
        (eag_t, residue) = self.compute_backward_ops(stage, eag_t, residue=residue)

        if self._monitor is not None:
            report = {
                'pass': 'backward',
                'stage': stage,
                'grad': {
                    'ea': eag_t
                },
                'residue': residue
            }
            self._monitor(report)

        if self.has_prev:
            return self.prev.backward(stage, eag_t, residue=residue)
        else:
            warnings.warn(f'Socket {self.name} connection is incomplete. Missing connection to previous layer.', UserWarning)
            return self

    @abc.abstractmethod
    def compute_forward_ops(self):
        """
        Compute the forwarded operation function. Not implemented.
        """
        pass

    @abc.abstractmethod
    def compute_backward_ops(self):
        """
        Compute the backwarded operation function. Not implemented.
        """
        pass


# ------------------------------------------------------------------------


class Dropout(Socket):
    _label = SOCKET.DROPOUT_LABEL
    """
    A dropout socket class.
    Arguments:
        size:
        name:
        pzero: dropping probability
    """
    @MType(size=int,
           name=str,
           pzero=float)
    def __init__(self, *,
                 size=1,
                 name='',
                 pzero=SOCKET.DEFAULT_DROPOUT_PZERO):
        self._pzero = SOCKET.DEFAULT_DROPOUT_PZERO
        self._mask_init = RandomBinary()
        self._mask_t = None
        self._pzero_scheduler = None

        super().__init__(shape=(1, size), name=name)
        self.reconfig(pzero=pzero)

    # ------------------------------------------------------------------------

    def reset(self):
        """
        Reset internal evaluation states
        """
        super().reset()
        self._mask_t = None

    def unassign_hooks(self):
        """
        Unassign all callback functions
        """
        super().unassign_hooks()
        self._pzero_scheduler = None

    @MType(monitor=OneOfType(callable, None),
           pzero_scheduler=OneOfType(callable, None))
    def assign_hook(self, *,
                    monitor=None,
                    pzero_scheduler=None):
        """
        Assign callback functions
        Arguments:
            monitor:
            pzero_scheduler: callback function to schedule the pzero
        """
        super().assign_hook(monitor=monitor)
        if pzero_scheduler is not None:
            self._pzero_scheduler = pzero_scheduler

    @MType(shape=OneOfType((int,), None),
           pzero=OneOfType(float, None))
    def reconfig(self, *,
                 shape=None,
                 pzero=None):
        """
        Reconfig dropout.
        Arguments:
            shape:
            pzero:
        """
        if pzero is not None:
            if pzero < 0 or pzero >= 1:
                warnings.warn(f'Dropout probability cannot be < 0 or >= 1. Reset to {SOCKET.DEFAULT_DROPOUT_PZERO}.', UserWarning)
                pzero = SOCKET.DEFAULT_DROPOUT_PZERO
            self._pzero = pzero
        if shape is not None:
            super().reconfig(shape=shape)
        self.reset()

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return dropout as a snapshot dict data
        Arguments:
            as_json -
            beautify_json -
        Returns:
            dict
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'pzero':  self._pzero
            # 'mask': {
            #     'dtype': str(self._mask_t.dtype),
            #     'values': self._mask_t.tolist()
            # } if self._mask_t is not None else None,
            # 'mask_init': self._mask_init.snapshot(as_json=False, beautify_json=False),
        })
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()

    @MType(int)
    def compute_pzero(self, epoch):
        """
        Get current regularization rate.
        Arguments:
            epoch:
        Returns:
            float
        """
        pzero = self._pzero
        if self._pzero_scheduler is not None:
            pzero = self._pzero_scheduler(epoch, self._pzero)
            if not isinstance(pzero, float) or pzero < 0:
                raise TypeError('Dropout propability value must be a positive floating point number.')
        return pzero

    @MType(dict, np.ndarray, residue=dict)
    @MShape(axis=1)
    def compute_forward_ops(self, stage, a_t, *, residue={}):
        """
        Do a dropout forwarded operation function on the post-nonlinear (a) tensor and residue.
        Arguments:
            stage: forward stage:
            a_t: post-nonlinearity (a) tensor
            residue:
        Returns:
            tensor
        """
        if 'epoch' in stage:
            epoch = stage['epoch']
        else:
            raise ValueError('Input stage is missing the required epoch number.')

        pzero = self.compute_pzero(epoch)
        if pzero > 0:
            self._mask_t = self._mask_init((1, self.size), pzero=pzero, dtype=np.int8)
        if self._mask_t is not None:
            a_t = np.multiply(a_t, self._mask_t)
        return (a_t, residue)

    @MType(dict, np.ndarray, residue=dict)
    @MShape(axis=1)
    def compute_backward_ops(self, stage, eag_t, *, residue={}):
        """
        Do a dropout backwarded operation function on gradient post-nonlinear (a) tensor and residue.
        Arguments:
            stage: backward stage
            eag_t: gradient error tensor w.r.t. post-nonlinearity (a) tensor
            residue:
        Returns:
            tensor
        """
        if self._mask_t is not None:
            eag_t = np.multiply(eag_t, self._mask_t)

        return (eag_t, residue)


# ------------------------------------------------------------------------


class BatchNorm(Socket):
    _label = SOCKET.BATCH_NORMALIZER_LABEL
    """
    Arguments:
        size: normalizer size
        name: normalizer name
        moving_mean_init:
        moving_variance_init:
        gamma_init:
        beta_init:
        optim:
    """
    @MType(size=int,
           name=str,
           moving_mean_init=OneOfType(str, float, Initializer),
           moving_variance_init=OneOfType(str, float, Initializer),
           gamma_init=OneOfType(str, float, Initializer),
           beta_init=OneOfType(str, float, Initializer),
           optim=OneOfType(str, Optimizer))
    def __init__(self, *,
                 size=1,
                 name='',
                 moving_mean_init='zeros',
                 moving_variance_init='ones',
                 gamma_init='ones',
                 beta_init='zeros',
                 optim='sgdm'):
        self._frozen = False
        self._optim = None

        self._a_hat_t = None
        self._a_offset_t = None

        self._mean_v = None
        self._variance_v = None

        self._moving_mean_init = None
        self._moving_variance_init = None
        self._moving_mean_v = None
        self._moving_variance_v = None

        self._gamma_v = None
        self._beta_v = None
        self._gamma_init = None
        self._beta_init = None

        super().__init__(shape=(1, size), name=name)
        self.reconfig(moving_mean_init=moving_mean_init,
                      moving_variance_init=moving_variance_init,
                      gamma_init=gamma_init,
                      beta_init=beta_init,
                      optim=optim)

    # ------------------------------------------------------------------------

    @property
    def is_frozen(self):
        """
        Check if layer is frozen
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
    def optim(self):
        """
        Get normalizer optimizer
        Returns:
            optimizer
        """
        return self._optim

    @property
    def moving_means(self):
        """
        Get normalizer moving mean vector
        Returns:
            moving mean vector
        """
        if self._moving_mean_v is not None:
            return self._moving_mean_v.copy()
        else:
            return None

    @moving_means.setter
    @MType(np.ndarray)
    @MShape(axis=-1)
    def moving_means(self, moving_mean_v):
        """
        Set normalizer moving mean vector
        """
        if self.is_frozen:
            warnings.warn(f'Cannot set moving means to a frozen normalizer {self.name}.', UserWarning)
        else:
            np.copyto(self._moving_mean_v, moving_mean_v, casting='same_kind')

    @property
    def moving_variances(self):
        """
        Get normalizer moving variance vector
        Returns:
            moving variance vector
        """
        if self._moving_variance_v is not None:
            return self._moving_variance_v.copy()
        else:
            return None

    @moving_variances.setter
    @MType(np.ndarray)
    @MShape(axis=-1)
    def moving_variances(self, moving_variance_v):
        """
        Set normalizer moving variance vector
        """
        if self.is_frozen:
            warnings.warn(f'Cannot set moving variances to a frozen normalizer {self.name}.', UserWarning)
        else:
            np.copyto(self._moving_variance_v, moving_variance_v, casting='same_kind')

    @property
    def gammas(self):
        """
        Get normalizer gamma vector
        Returns:
            gamma vector
        """
        if self._gamma_v is not None:
            return self._gamma_v.copy()
        else:
            return None

    @gammas.setter
    @MType(np.ndarray)
    @MShape(axis=-1)
    def gammas(self, gamma_v):
        """
        Set normalizer gamma vector
        """
        if self.is_frozen:
            warnings.warn(f'Cannot set gammas to a frozen normalizer {self.name}.', UserWarning)
        else:
            np.copyto(self._gamma_v, gamma_v, casting='same_kind')

    @property
    def betas(self):
        """
        Get normalizer beta vector
        Returns:
            beta vector
        """
        if self._beta_v is not None:
            return self._beta_v.copy()
        else:
            return None

    @betas.setter
    @MType(np.ndarray)
    @MShape(axis=-1)
    def betas(self, beta_v):
        """
        Set normalizer beta vector
        """
        if self.is_frozen:
            warnings.warn(f'Cannot set betas to a frozen normalizer {self.name}.', UserWarning)
        else:
            np.copyto(self._beta_v, beta_v, casting='same_kind')

    def unassign_hooks(self):
        """
        Unassign all callback functions
        """
        super().unassign_hooks()

    @MType(monitor=OneOfType(callable, None))
    def assign_hook(self, *,
                    monitor=None):
        """
        Assign callback functions
        Arguments:
            monitor: callback function to do probing during forward/backward pass
        """
        super().assign_hook(monitor=monitor)

    def reset(self):
        """
        Reset params to initial values
        """
        super().reset()
        self._a_hat_t = None
        self._a_offset_t = None

        self._mean_v = None
        self._variance_v = None

        if self._moving_mean_init is not None:
            self._moving_mean_v = self._moving_mean_init(self.shape)
        if self._moving_variance_init is not None:
            self._moving_variance_v = self._moving_variance_init(self.shape)
        if self._gamma_init is not None:
            self._gamma_v = self._gamma_init(self.shape)
        if self._beta_init is not None:
            self._beta_v = self._beta_init(self.shape)

        if self._optim is not None:
            self._optim.reset()

    @MType(shape=OneOfType((int,), None),
           moving_mean_init=OneOfType(str, float, Initializer, None),
           moving_variance_init=OneOfType(str, float, Initializer, None),
           gamma_init=OneOfType(str, float, Initializer, None),
           beta_init=OneOfType(str, float, Initializer, None),
           optim=OneOfType(str, Optimizer, None))
    def reconfig(self, *,
                 shape=None,
                 moving_mean_init=None,
                 moving_variance_init=None,
                 gamma_init=None,
                 beta_init=None,
                 optim=None):
        """
        Reconfig batch normalizer
        Arguments:
            shape:
            moving_mean_init:
            moving_variance_init:
            gamma_init:
            beta_init:
            optim:
        """
        if moving_mean_init is not None:
            if isinstance(moving_mean_init, str):
                moving_mean_init_label = moving_mean_init
                if self._moving_mean_init is not None and moving_mean_init_label == self._moving_mean_init.label:
                    warnings.warn(
                        'No change made to normalizer gamma. Re-initializing gamma skipped.',
                        UserWarning)
                else:
                    if Zeros.label == moving_mean_init_label:
                        self._moving_mean_init = Zeros()
                    elif Ones.label == moving_mean_init_label:
                        self._moving_mean_init = Ones()
                    elif RandomNormal.label == moving_mean_init_label:
                        self._moving_mean_init = RandomNormal()
                    elif RandomUniform.label == moving_mean_init_label:
                        self._moving_mean_init = RandomUniform()
                    elif GlorotRandomNormal.label == moving_mean_init_label:
                        self._moving_mean_init = GlorotRandomNormal()
                    elif GlorotRandomUniform.label == moving_mean_init_label:
                        self._moving_mean_init = GlorotRandomUniform()
                    else:
                        raise TypeError(f'Unknown moving mean initializer {moving_mean_init_label} for normalizer {self.name}.')
                    self._moving_mean_v = self._moving_mean_init(self.shape)
            elif isinstance(moving_mean_init, float):
                self._moving_mean_init = Constant(moving_mean_init)
                self._moving_mean_v = self._moving_mean_init(self.shape)
            else:
                if self._moving_mean_init is not None and moving_mean_init.label == self._moving_mean_init.label:
                    warnings.warn(
                        'No change made to normalizer moving mean initializer. Re-initializing moving means skipped.',
                        UserWarning)
                else:
                    self._moving_mean_init = moving_mean_init
                    self._moving_mean_v = self._moving_mean_init(self.shape)
        if moving_variance_init is not None:
            if isinstance(moving_variance_init, str):
                moving_variance_init_label = moving_variance_init
                if self._moving_variance_init is not None and moving_variance_init_label == self._moving_variance_init.label:
                    warnings.warn(
                        'No change made to normalizer gamma. Re-initializing gamma skipped.',
                        UserWarning)
                else:
                    if Zeros.label == moving_variance_init_label:
                        self._moving_variance_init = Zeros()
                    elif Ones.label == moving_variance_init_label:
                        self._moving_variance_init = Ones()
                    elif RandomNormal.label == moving_variance_init_label:
                        self._moving_variance_init = RandomNormal()
                    elif RandomUniform.label == moving_variance_init_label:
                        self._moving_variance_init = RandomUniform()
                    elif GlorotRandomNormal.label == moving_variance_init_label:
                        self._moving_variance_init = GlorotRandomNormal()
                    elif GlorotRandomUniform.label == moving_variance_init_label:
                        self._moving_variance_init = GlorotRandomUniform()
                    else:
                        raise TypeError(f'Unknown moving variance initializer {moving_variance_init_label} for normalizer {self.name}.')
                    self._moving_variance_v = self._moving_variance_init(self.shape)
            elif isinstance(moving_variance_init, float):
                self._moving_variance_init = Constant(moving_variance_init)
                self._moving_variance_v = self._moving_variance_init(self.shape)
            else:
                if self._moving_variance_init is not None and moving_variance_init.label == self._moving_variance_init.label:
                    warnings.warn(f'No change made to normalizer moving variance initializer. Re-initializing moving variances skipped.', UserWarning)
                else:
                    self._moving_variance_init = moving_variance_init
                    self._moving_variance_v = self._moving_variance_init(self.shape)
        if gamma_init is not None:
            if isinstance(gamma_init, str):
                gamma_init_label = gamma_init
                if self._gamma_init is not None and gamma_init_label == self._gamma_init.label:
                    warnings.warn(f'No change made to normalizer gamma initializer. Re-initializing gammas skipped.', UserWarning)
                else:
                    if Zeros.label == gamma_init_label:
                        self._gamma_init = Zeros()
                    elif Ones.label == gamma_init_label:
                        self._gamma_init = Ones()
                    elif RandomNormal.label == gamma_init_label:
                        self._gamma_init = RandomNormal()
                    elif RandomUniform.label == gamma_init_label:
                        self._gamma_init = RandomUniform()
                    elif GlorotRandomNormal.label == gamma_init_label:
                        self._gamma_init = GlorotRandomNormal()
                    elif GlorotRandomUniform.label == gamma_init_label:
                        self._gamma_init = GlorotRandomUniform()
                    else:
                        raise TypeError(f'Unknown gamma initializer {gamma_init_label} for normalizer {self.name}.')
                    self._gamma_v = self._gamma_init(self.shape)
            elif isinstance(gamma_init, float):
                self._gamma_init = Constant(gamma_init)
                self._gamma_v = self._gamma_init(self.shape)
            else:
                if self._gamma_init is not None and gamma_init.label == self._gamma_init.label:
                    warnings.warn(
                        'No change made to normalizer gamma initializer. Re-initializing gammas skipped.',
                        UserWarning)
                else:
                    self._gamma_init = gamma_init
                    self._gamma_v = self._gamma_init(self.shape)
        if beta_init is not None:
            if isinstance(beta_init, str):
                beta_init_label = beta_init
                if self._beta_init is not None and beta_init_label == self._beta_init.label:
                    warnings.warn(
                        'No change made to normalizer beta initializer. Re-initializing betas skipped.',
                        UserWarning)
                else:
                    if Zeros.label == beta_init_label:
                        self._beta_init = Zeros()
                    elif Ones.label == beta_init_label:
                        self._beta_init = Ones()
                    elif RandomNormal.label == beta_init_label:
                        self._beta_init = RandomNormal()
                    elif RandomUniform.label == beta_init_label:
                        self._beta_init = RandomUniform()
                    elif GlorotRandomNormal.label == beta_init_label:
                        self._beta_init = GlorotRandomNormal()
                    elif GlorotRandomUniform.label == beta_init_label:
                        self._beta_init = GlorotRandomUniform()
                    else:
                        raise TypeError(f'Unknown beta initializer {beta_init_label} for normalizer {self.name}.')
                    self._beta_v = self._beta_init(self.shape)
            elif isinstance(beta_init, float):
                self._beta_init = Constant(beta_init)
                self._beta_v = self._beta_init(self.shape)
            else:
                if self._beta_init is not None and beta_init.label == self._beta_init.label:
                    warnings.warn(
                        'No change made to normalizer beta initializer. Re-initializing betas skipped.',
                        UserWarning)
                else:
                    self._beta_init = beta_init
                    self._beta_v = self._beta_init(self.shape)
        if optim is not None:
            if isinstance(optim, str):
                optim_label = optim
                if self._optim is not None and optim_label == self._optim.label:
                    warnings.warn(
                        'No change made to normalizer optimizer. Reconfig normalizer optimization skipped.',
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
                        raise TypeError(f'Unknown optimizer {optim_label} for normalizer {self.name}.')
            else:
                if self._optim is not None and optim.label == self._optim.label:
                    warnings.warn(
                        'No change made to normalizer. Reconfig normalizer optimization skipped.',
                        UserWarning)
                else:
                    self._optim = optim
        if shape is not None:
            super().reconfig(shape=shape)
        self.reset()

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return normalizer as a snapshot dict data
        Arguments:
            as_json:
            beautify_json:
        Returns:
            snapshot
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'moving_mean': {
                'dtype': str(self.moving_means.dtype),
                'values': self.moving_means.tolist()
            } if self.moving_means is not None else None,
            'moving_variance': {
                'dtype': str(self.moving_variances.dtype),
                'values': self.moving_variances.tolist()
            } if self.moving_variances is not None else None,
            'gamma': {
                'dtype': str(self.gammas.dtype),
                'values': self.gammas.tolist()
            } if self.gammas is not None else None,
            'beta': {
                'dtype': str(self.betas.dtype),
                'values': self.betas.tolist()
            } if self.betas is not None else None,
            'moving_mean_init': self._moving_mean_init.snapshot(as_json=False, beautify_json=False),
            'moving_variance_init': self._moving_variance_init.snapshot(as_json=False, beautify_json=False),
            'gamma_init': self._gamma_init.snapshot(as_json=False, beautify_json=False),
            'beta_init': self._beta_init.snapshot(as_json=False, beautify_json=False),
            'optim': self._optim.snapshot(as_json=False, beautify_json=False)
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
    def compute_forward_ops(self, stage, a_t, *, residue={}):
        """
        Do a dropout forwarded operation function on the post-nonlinear (a) tensor and residue.
        Arguments:
            stage: forward stage
            a_t: post-nonlinearity (a) tensor
            residue:
        Returns:
            tensor
        """
        mode = stage['mode']
        if mode == 'learning' or mode == 'infering':
            self._mean_v = np.mean(a_t, axis=0)
            self._variance_v = np.mean(np.square(a_t - self._mean_v), axis=0)
            self._a_hat_t = (a_t - self._mean_v) / np.sqrt(self._variance_v + 1e-12)
            self._a_offset_t = a_t - self._mean_v
            a_t = (self._gamma_v * self._a_hat_t) + self._beta_v

            self._moving_mean_v = SOCKET.DEFAULT_BATCH_NORMALIZER_MOVING_MOMENTUM * self._moving_mean_v + (1 - SOCKET.DEFAULT_BATCH_NORMALIZER_MOVING_MOMENTUM) * self._mean_v
            self._moving_variance_v = SOCKET.DEFAULT_BATCH_NORMALIZER_MOVING_MOMENTUM * self._moving_variance_v + (1 - SOCKET.DEFAULT_BATCH_NORMALIZER_MOVING_MOMENTUM) * self._variance_v
            self._moving_mean_v = self._moving_mean_v.astype(np.float32)
            self._moving_variance_v = self._moving_variance_v.astype(np.float32)
        else:
            self._a_hat_t = (a_t - self._moving_mean_v) / np.sqrt(self._moving_variance_v + 1e-12)
            a_t = (self._gamma_v * self._a_hat_t) + self._beta_v
        return (a_t, residue)

    @MType(dict, np.ndarray, dict, residue=dict)
    @MShape(axis=1)
    def compute_backward_ops(self, stage, eag_t, *, residue={}):
        """
        Do a dropout backwarded operation function on gradient post-nonlinear (a) tensor and residue.
        Arguments:
            stage: backward stage
            eag_t: gradient error tensor w.r.t. post-nonlinearity (a) tensor
            residue:
        Returns:
            tensor
        """
        epoch = stage['epoch']
        mode = stage['mode']
        hparam = stage['hparam']
        batch_size = hparam['batch_size']
        if mode == 'learning' or mode == 'infering':
            gammag_v = np.sum(self._a_offset_t * (self._variance_v + 1e-12)**(-0.5) * eag_t, axis=0)
            betag_v = np.sum(eag_t, axis=0)

            [gamma_delta_v, beta_delta_v] = self._optim.compute_grad_descent_step(epoch, [gammag_v, betag_v], hparam)
            self._gamma_v -= gamma_delta_v
            self._beta_v -= beta_delta_v

            if batch_size == 1:
                eag_t = self._gamma_v * (eag_t - np.sum(eag_t, axis=0) - (self._a_offset_t * np.sum(eag_t * self._a_offset_t, axis=0)) / (self._variance_v + 1e-12))
                eag_t = eag_t / np.sqrt(self._variance_v + 1e-12)
            else:
                eag_t = self._gamma_v * (batch_size * eag_t - np.sum(eag_t, axis=0) - (self._a_offset_t * np.sum(eag_t * self._a_offset_t, axis=0)) / (self._variance_v + 1e-12))
                eag_t = eag_t / (batch_size * np.sqrt(self._variance_v + 1e-12))

        return (eag_t, residue)
