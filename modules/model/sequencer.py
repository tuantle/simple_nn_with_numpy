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

import warnings
import json
import numpy as np

from util.const import CONST
from util.validation import (
    FType,
    MType,
    OneOfType
)
from npcore.layer.link import Link
from npcore.layer.gates import (
    Gate,
    Linear,
    ReLU,
    LeakyReLU,
    ELU,
    Swish,
    SoftPlus,
    Sigmoid,
    Tanh,
    Algebraic
)
from npcore.layer.sockets import (
    Socket,
    Dropout,
    BatchNorm,
)
from npcore.optimizers import (
    Optimizer,
    SGD,
    SGDM,
    RMSprop,
    Adam
)
from npcore.initializers import (
    Initializer,
    Zeros, Ones, Constant,
    Identity, Diagonal,
    RandomNormal, RandomUniform,
    GlorotRandomNormal, GlorotRandomUniform
)
from npcore.regularizers import (
    Regularizer,
    L1Lasso,
    L2Ridge,
    L1L2ElasticNet
)


# ------------------------------------------------------------------------


class SEQUENCER(CONST):
    LABEL = 'sequencer'


# ------------------------------------------------------------------------


class Sequencer(type):
    """
    A metaclass for a base sequencer class
    """
    def __getattr__(cls, key):
        return cls[key]

    def __setattr__(cls, key, value):
        raise ValueError('Cannot set value to a sequencer class.')

    @property
    def label(cls):
        """
        Get sequencer label.
        Returns:
            str
        """
        return cls._label


# ------------------------------------------------------------------------


class Sequencer(object, metaclass=Sequencer):
    _label = SEQUENCER.LABEL
    """
    A sequencer class.
    Arguments:
        name:
    """

    @MType(name=str)
    def __init__(self, *,
                 name=''):
        self._name = name
        self._valid_sequence = False
        self._sequence = None
        # self._registry = {}

    def __str__(self):
        if self.name != '':
            return self.name + '_' + self.label
        else:
            return self.label

    # ------------------------------------------------------------------------

    @property
    def label(self):
        """
        Get layer label.
        Returns:
            str
        """
        return type(self).label

    @property
    def name(self):
        """
        Get sequencer name
        Returns:
        """
        return self._name

    @name.setter
    @MType(str)
    def name(self, name):
        """
        Set sequencer name
        Arguments:
            name: sequencer name
        """
        self._name = name

    @property
    def sequence(self):
        """
        Get sequencer sequence
        Returns:
        """
        if self.is_valid:
            return self._sequence
        else:
            return None

    @property
    def is_valid(self):
        """
        Check that sequence is valid
        Returns:
        """
        return self._sequence is not None

    @property
    def is_complete(self):
        """
        Check that sequence is complete
        Returns:
        """
        return self.is_valid and self._valid_sequence

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return sequencer as a snapshot dict data
        Arguments:
            as_json:
            beautify_json:
        Returns:
            dict
        """
        sequencer_snapshot = {
            'name': self.name,
            'label': self.label,
            'base_label': Sequencer.label,
            'sequences': []
        }
        if self.is_complete:
            for layer in self.sequence.head:
                sequencer_snapshot['sequences'].append(layer.snapshot(as_json=False, beautify_json=False))

        if as_json:
            if beautify_json:
                return json.dumps(sequencer_snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(sequencer_snapshot)
        else:
            return sequencer_snapshot.copy()

    @MType(dict, overwrite=bool)
    def load_snapshot(self, sequencer_snapshot, *, overwrite=False):
        """
        Load sequence from file
        Arguments:
            sequencer_snapshot:
            overwrite:
        Returns:
            self
        """
        if self.is_valid and not overwrite:
            raise RuntimeError(f'Sequencer {self.name} sequence is valid. Cannot overwrite sequence.')
        sequence = None
        for sequence_snapshot in sequencer_snapshot['sequences'][:-1]:
            layer_label = sequence_snapshot['label']
            name = sequence_snapshot['name']
            shape = tuple(sequence_snapshot['shape'])
            size = shape[1]

            if Linear.label == layer_label:
                layer = Linear(size=size, name=name)
            elif ReLU.label == layer_label:
                layer = ReLU(size=size, name=name)
            elif LeakyReLU.label == layer_label:
                layer = LeakyReLU(size=size, name=name)
            elif ELU.label == layer_label:
                layer = ELU(size=size, name=name)
            elif SoftPlus.label == layer_label:
                layer = SoftPlus(size=size, name=name)
            elif Swish.label == layer_label:
                layer = Swish(size=size, name=name)
            elif Sigmoid.label == layer_label:
                layer = Sigmoid(size=size, name=name)
            elif Tanh.label == layer_label:
                layer = Tanh(size=size, name=name)
            elif Algebraic.label == layer_label:
                layer = Algebraic(size=size, name=name)
            elif Dropout.label == layer_label:
                pzero = sequence_snapshot['pzero']
                layer = Dropout(size=size, name=name, pzero=pzero)
            elif BatchNorm.label == layer_label:
                optim = 'sgdm'
                optim_label = sequence_snapshot['optim']['label']
                if SGD.label == optim_label:
                    optim = SGD()
                elif SGDM.label == optim_label:
                    optim = SGDM()
                elif RMSprop.label == optim_label:
                    optim = RMSprop()
                elif Adam.label == optim_label:
                    optim = Adam()
                else:
                    raise TypeError(f'Unknown optimizer {optim_label} for normalizer {name}.')

                moving_mean_init = 'zeros'
                moving_mean_init_label = sequence_snapshot['moving_mean_init']['label']
                if Zeros.label == moving_mean_init_label:
                    moving_mean_init = Zeros()
                elif Ones.label == moving_mean_init_label:
                    moving_mean_init = Ones()
                elif RandomNormal.label == moving_mean_init_label:
                    moving_mean_init = RandomNormal()
                elif RandomUniform.label == moving_mean_init_label:
                    moving_mean_init = RandomUniform()
                elif GlorotRandomNormal.label == moving_mean_init_label:
                    moving_mean_init = GlorotRandomNormal()
                elif GlorotRandomUniform.label == moving_mean_init_label:
                    moving_mean_init = GlorotRandomUniform()
                else:
                    raise TypeError(f'Unknown moving mean initializer {moving_mean_init_label} for normalizer {name}.')
                moving_variance_init = 'ones'
                moving_variance_init_label = sequence_snapshot['moving_variance_init']['label']
                if Zeros.label == moving_variance_init_label:
                    moving_variance_init = Zeros()
                elif Ones.label == moving_variance_init_label:
                    moving_variance_init = Ones()
                elif RandomNormal.label == moving_variance_init_label:
                    moving_variance_init = RandomNormal()
                elif RandomUniform.label == moving_variance_init_label:
                    moving_variance_init = RandomUniform()
                elif GlorotRandomNormal.label == moving_variance_init_label:
                    moving_variance_init = GlorotRandomNormal()
                elif GlorotRandomUniform.label == moving_variance_init_label:
                    moving_variance_init = GlorotRandomUniform()
                else:
                    raise TypeError(f'Unknown moving variance initializer {moving_variance_init_label} for normalizer {name}.')

                gamma_init = 'ones'
                gamma_init_label = sequence_snapshot['gamma_init']['label']
                if Zeros.label == gamma_init_label:
                    gamma_init = Zeros()
                elif Ones.label == gamma_init_label:
                    gamma_init = Ones()
                elif RandomNormal.label == gamma_init_label:
                    gamma_init = RandomNormal()
                elif RandomUniform.label == gamma_init_label:
                    gamma_init = RandomUniform()
                elif GlorotRandomNormal.label == gamma_init_label:
                    gamma_init = GlorotRandomNormal()
                elif GlorotRandomUniform.label == gamma_init_label:
                    gamma_init = GlorotRandomUniform()
                else:
                    raise TypeError(f'Unknown gamma initializer {gamma_init_label} for normalizer {name}.')
                beta_init = 'zeros'
                beta_init_label = sequence_snapshot['beta_init']['label']
                if Zeros.label == beta_init_label:
                    beta_init = Zeros()
                elif Ones.label == beta_init_label:
                    beta_init = Ones()
                elif RandomNormal.label == beta_init_label:
                    beta_init = RandomNormal()
                elif RandomUniform.label == beta_init_label:
                    beta_init = RandomUniform()
                elif GlorotRandomNormal.label == beta_init_label:
                    beta_init = GlorotRandomNormal()
                elif GlorotRandomUniform.label == beta_init_label:
                    beta_init = GlorotRandomUniform()
                else:
                    raise TypeError(f'Unknown beta initializer {beta_init_label} for normalizer {name}.')
                layer = BatchNorm(size=size,
                                  name=name,
                                  moving_mean_init=moving_mean_init,
                                  moving_variance_init=moving_variance_init,
                                  gamma_init=gamma_init,
                                  beta_init=beta_init,
                                  optim=optim)
                layer.moving_means = np.array(sequence_snapshot['moving_mean']['values'], dtype=sequence_snapshot['moving_mean']['dtype'])
                layer.moving_variances = np.array(sequence_snapshot['moving_variance']['values'], dtype=sequence_snapshot['moving_variance']['dtype'])
                layer.gammas = np.array(sequence_snapshot['gamma']['values'], dtype=sequence_snapshot['gamma']['dtype'])
                layer.betas = np.array(sequence_snapshot['beta']['values'], dtype=sequence_snapshot['beta']['dtype'])
            elif Link.label == layer_label:
                frozen = sequence_snapshot['frozen']
                weight_init = 'random_normal'
                weight_init_label = sequence_snapshot['weight_init']['label']
                if Zeros.label == weight_init_label:
                    weight_init = Zeros()
                elif Ones.label == weight_init_label:
                    weight_init = Ones()
                elif Identity.label == weight_init_label:
                    weight_init = Identity()
                elif Diagonal.label == weight_init_label:
                    value = sequence_snapshot['weight_init']['value']
                    weight_init = Diagonal(value)
                elif RandomNormal.label == weight_init_label:
                    seed = sequence_snapshot['weight_init']['seed']
                    mean = sequence_snapshot['weight_init']['mean']
                    variance = sequence_snapshot['weight_init']['variance']
                    weight_init = RandomNormal(seed=seed,
                                               mean=mean,
                                               variance=variance)
                elif RandomUniform.label == weight_init_label:
                    seed = sequence_snapshot['weight_init']['seed']
                    min = sequence_snapshot['weight_init']['min']
                    max = sequence_snapshot['weight_init']['max']
                    weight_init = RandomUniform(seed=seed,
                                                min=min,
                                                max=max)
                elif GlorotRandomNormal.label == weight_init_label:
                    seed = sequence_snapshot['weight_init']['seed']
                    weight_init = GlorotRandomNormal(seed=seed)
                elif GlorotRandomUniform.label == weight_init_label:
                    seed = sequence_snapshot['weight_init']['seed']
                    weight_init = GlorotRandomUniform(seed=seed)
                else:
                    raise TypeError(f'Unknown weight initializer {weight_init_label} for link {name}.')

                weight_reg = 'not_use'
                if sequence_snapshot['weight_reg'] is not None:
                    weight_reg_label = sequence_snapshot['weight_reg']['label']
                    if L1Lasso.label == weight_reg_label:
                        weight_reg = L1Lasso()
                    elif L2Ridge.label == weight_reg_label:
                        weight_reg = L2Ridge()
                    elif L1L2ElasticNet.label == weight_reg_label:
                        weight_reg = L1L2ElasticNet()
                    else:
                        raise TypeError(f'Unknown weight regularizer {weight_reg_label} for link {name}.')

                bias_init = 'not_use'
                if BatchNorm.label == sequence.tail.label and bias_init != 'not_use':
                    warnings.warn(f'Link biases is not needed with batch normalization in the previous layer enabled. Link biases initialization skipped.', UserWarning)
                else:
                    if sequence_snapshot['bias_init'] is not None:
                        bias_init_label = sequence_snapshot['bias_init']['label']
                        if Zeros.label == bias_init_label:
                            bias_init = Zeros()
                        elif Ones.label == bias_init_label:
                            bias_init = Ones()
                        elif Constant.label == bias_init_label:
                            value = sequence_snapshot['bias_init']['value']
                            bias_init = Constant(value)
                        else:
                            raise TypeError(f'Unknown bias initializer {bias_init_label} for link {name}.')

                optim = 'sgd'
                optim_label = sequence_snapshot['optim']['label']
                if SGD.label == optim_label:
                    optim = SGD()
                elif SGDM.label == optim_label:
                    optim = SGDM()
                elif RMSprop.label == optim_label:
                    optim = RMSprop()
                elif Adam.label == optim_label:
                    optim = Adam()
                else:
                    raise TypeError(f'Unknown optimizer {optim_label} for link {name}.')

                layer = Link(shape=shape,
                             name=name,
                             weight_init=weight_init,
                             weight_reg=weight_reg,
                             bias_init=bias_init,
                             optim=optim)

                layer.weights = np.array(sequence_snapshot['weight']['values'], dtype=sequence_snapshot['weight']['dtype'])
                if sequence_snapshot['bias'] is not None:
                    layer.biases = np.array(sequence_snapshot['bias']['values'], dtype=sequence_snapshot['bias']['dtype'])

                if frozen:
                    layer.freeze()

            if sequence is None:
                sequence = layer
            else:
                sequence.tail.connect(layer)

        self._name = sequencer_snapshot['name']
        self._sequence = sequence
        self._valid_sequence = True

        return self

    @classmethod
    @MType(OneOfType(str, Gate, Socket),
           size=OneOfType(int, None),
           shape=OneOfType((int,), None),
           name=str)
    def create(cls, layer, *,
               size=None,
               shape=None,
               name=''):
        """
        Create a sequencer with layers.
        Arguments:
            size:
            shape:
            layer:
            name:
        Returns:
            callable
        """
        @FType(OneOfType(callable, Sequencer, None))
        def connect(preceded_sequencer):
            """
            Connect new layer to preceded sequencer sequence.
            Arguments:
                preceded_sequencer:
            Returns:
                sequencer
            """
            nonlocal layer
            nonlocal size

            if preceded_sequencer is None:
                preceded_sequencer = cls(name=name)
            elif callable(preceded_sequencer):
                preceded_sequencer = preceded_sequencer(None)

            sequence = None
            if isinstance(layer, str):
                layer_label = layer
                if size is None:
                    if preceded_sequencer.is_valid:
                        prev_layer_size = preceded_sequencer.sequence.tail.size
                        size = prev_layer_size
                        layer.reconfig(shape=(1, size))
                    else:
                        warnings.warn('Gate layer size is not specified. Using size = 1.', UserWarning)
                        size = 1

                if Linear.label == layer_label:
                    layer = Linear(size=size, name=name)
                elif ReLU.label == layer_label:
                    layer = ReLU(size=size, name=name)
                elif LeakyReLU.label == layer_label:
                    layer = LeakyReLU(size=size, name=name)
                elif ELU.label == layer_label:
                    layer = ELU(size=size, name=name)
                elif SoftPlus.label == layer_label:
                    layer = SoftPlus(size=size, name=name)
                elif Swish.label == layer_label:
                    layer = Swish(size=size, name=name)
                elif Sigmoid.label == layer_label:
                    layer = Sigmoid(size=size, name=name)
                elif Tanh.label == layer_label:
                    layer = Tanh(size=size, name=name)
                elif Algebraic.label == layer_label:
                    layer = Algebraic(size=size, name=name)
                else:
                    raise TypeError(f'Unknown gate layer label {layer_label}.')

                if preceded_sequencer.is_valid:
                    prev_layer_label = preceded_sequencer.sequence.tail.label
                    prev_layer_size = preceded_sequencer.sequence.tail.size
                    shape = (prev_layer_size, size)
                    sequence = Link(shape=shape,
                                    name=name,
                                    weight_init='random_normal',
                                    weight_reg='not_use',
                                    bias_init='zeros' if BatchNorm.label != prev_layer_label else 'not_use',
                                    optim='sgd').connect(layer)
                else:
                    sequence = layer
            elif isinstance(layer, Gate):
                if size is None:
                    if preceded_sequencer.is_valid:
                        prev_layer_size = preceded_sequencer.sequence.tail.size
                        size = prev_layer_size
                        layer.reconfig(shape=(1, size))
                else:
                    if size != layer.size:
                        layer.reconfig(shape=(1, size))
                if name != '':
                    layer.name = name
                if preceded_sequencer.is_valid:
                    prev_layer_label = preceded_sequencer.sequence.tail.label
                    prev_layer_size = preceded_sequencer.sequence.tail.size
                    shape = (prev_layer_size, size)
                    sequence = Link(shape=shape,
                                    name=name,
                                    weight_init='random_normal',
                                    weight_reg='not_use',
                                    bias_init='zeros' if BatchNorm.label != prev_layer_label else 'not_use',
                                    optim='sgd').connect(layer)
                else:
                    sequence = layer
            elif isinstance(layer, Socket):
                if not preceded_sequencer.is_valid:
                    raise RuntimeError(f'Socket layer {layer_label} cannot be the first layer in sequence.')
                if size is None:
                    if preceded_sequencer.is_valid:
                        prev_layer_size = preceded_sequencer.sequence.tail.size
                        size = prev_layer_size
                        layer.reconfig(shape=(1, size))
                else:
                    if size != layer.size:
                        layer.reconfig(shape=(1, size))
                if name != '':
                    layer.name = name
                sequence = layer

            if preceded_sequencer.is_valid:
                preceded_sequencer.sequence.tail.connect(sequence.head)
            else:
                preceded_sequencer._sequence = sequence

            if preceded_sequencer.sequence.is_singular:
                preceded_sequencer._valid_sequence = False
            else:
                if Gate.label in str(preceded_sequencer.sequence.head) and \
                   Gate.label in str(preceded_sequencer.sequence.tail) and \
                   Link.label in str(preceded_sequencer.sequence.tail.prev):
                    preceded_sequencer._valid_sequence = True

            return preceded_sequencer
        return connect

    @MType(OneOfType(str, Gate, Socket),
           size=OneOfType(int, None),
           name=str)
    def add(self, layer, *,
            size=None,
            name=''):
        """
        Add new sequence layer
        Arguments:
            size:
            layer:
            name
        Returns:
            self
        """
        sequencer = self.create(layer,
                                size=size,
                                name=name)(self)
        self._sequence = sequencer.sequence
        self._valid_sequence = sequencer._valid_sequence
        return self

    @MType(pzero=OneOfType(float, None),
           weight_init=OneOfType(str, Initializer, None),
           weight_reg=OneOfType(str, Regularizer, None),
           bias_init=OneOfType(str, float, Initializer, None),
           moving_mean_init=OneOfType(str, float, Initializer, None),
           moving_variance_init=OneOfType(str, float, Initializer, None),
           gamma_init=OneOfType(str, float, Initializer, None),
           beta_init=OneOfType(str, float, Initializer, None),
           optim=OneOfType(str, Optimizer, None))
    def reconfig(self, *,
                 pzero=None,
                 weight_init=None,
                 weight_reg=None,
                 bias_init=None,
                 moving_mean_init=None,
                 moving_variance_init=None,
                 gamma_init=None,
                 beta_init=None,
                 optim=None):
        """
        Reconfig the previous layer in sequence.
        Arguments:
            pzero:
            weight_init:
            weight_reg:
            bias_init:
            moving_mean_init:
            moving_variance_init:
            gamma_init:
            beta_init:
            optim:
        Returns:
            self
        """
        if not self.is_valid:
            raise RuntimeError(f'Sequencer {self.name} sequence is valid.')
        layer = self.sequence.tail
        if Gate.label in str(layer):
            if layer.has_prev:
                if weight_init is None and weight_reg is None and \
                   bias_init is None and optim is None:
                    warnings.warn(f'No reconfiguration was applied to layer {layer.label}.', UserWarning)
                else:
                    layer.prev.reconfig(weight_init=weight_init,
                                        weight_reg=weight_reg,
                                        bias_init=bias_init,
                                        optim=optim)
            else:
                warnings.warn(f'Reconfiguration was applied. Layer {layer.label} reconfiguration skipped.', UserWarning)
        elif Dropout.label == layer.label:
            if pzero is None:
                warnings.warn(f'No reconfiguration was applied to layer {layer.label}.', UserWarning)
            else:
                layer.reconfig(pzero=pzero)
        elif BatchNorm.label == layer.label:
            if moving_mean_init is None and moving_variance_init is None and \
               gamma_init is None and beta_init is None and optim is None:
                warnings.warn(f'No reconfiguration was applied to layer {layer.label}.', UserWarning)
            else:
                layer.reconfig(moving_mean_init=moving_mean_init,
                               moving_variance_init=moving_variance_init,
                               gamma_init=gamma_init,
                               beta_init=beta_init,
                               optim=optim)
        return self

    @MType(pzero=OneOfType(float, None),
           weight_init=OneOfType(str, Initializer, None),
           weight_reg=OneOfType(str, Regularizer, None),
           bias_init=OneOfType(str, float, Initializer, None),
           moving_mean_init=OneOfType(str, float, Initializer, None),
           moving_variance_init=OneOfType(str, float, Initializer, None),
           gamma_init=OneOfType(str, float, Initializer, None),
           beta_init=OneOfType(str, float, Initializer, None),
           optim=OneOfType(str, Optimizer, None))
    def reconfig_all(self, *,
                     pzero=None,
                     weight_init=None,
                     weight_reg=None,
                     bias_init=None,
                     moving_mean_init=None,
                     moving_variance_init=None,
                     gamma_init=None,
                     beta_init=None,
                     optim=None):
        """
        Reconfig all previous layers in sequence.
        Arguments:
            pzero:
            weight_init:
            weight_reg:
            bias_init:
            moving_mean_init:
            moving_variance_init:
            gamma_init:
            beta_init:
            optim:
        Returns:
            self
        """
        if not self.is_valid:
            raise RuntimeError(f'Sequencer {self.name} sequence is valid.')
        for layer in self.sequence.head:
            if Link.label == layer.label:
                layer.reconfig(weight_init=weight_init,
                               weight_reg=weight_reg,
                               bias_init=bias_init,
                               optim=optim)
            elif Dropout.label == layer.label:
                layer.reconfig(pzero=pzero)
            elif BatchNorm.label == layer.label:
                layer.reconfig(moving_mean_init=moving_mean_init,
                               moving_variance_init=moving_variance_init,
                               gamma_init=gamma_init,
                               beta_init=beta_init,
                               optim=optim)
        return self
