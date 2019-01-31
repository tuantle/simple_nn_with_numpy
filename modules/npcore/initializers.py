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
import math
import numpy as np

from util.const import CONST
from util.validation import (
    MType,
    OneOfType
)


# ------------------------------------------------------------------------


class INITIALIZER(CONST):
    LABEL = 'init'
    ZEROS_LABEL = 'zeros'
    ONES_LABEL = 'ones'
    CONSTANT_LABEL = 'constant'
    IDENTITY_LABEL = 'identity'
    DIAGONAL_LABEL = 'diagonal'
    RANDOM_BINARY_LABEL = 'random_binary'
    RANDOM_ORTHONORMAL_LABEL = 'random_orthonormal'
    RANDOM_NORMAL_LABEL = 'random_normal'
    RANDOM_UNIFORM_LABEL = 'random_uniform'
    GLOROT_RANDOM_NORMAL_LABEL = 'glorot_random_normal'
    GLOROT_RANDOM_UNIFORM_LABEL = 'glorot_random_uniform'

    DEFAULT_RANDOM_NORMAL_MEAN = 0.0
    DEFAULT_RANDOM_NORMAL_VARIANCE = 1e-3
    DEFAULT_RANDOM_UNIFORM_MIN = -1e-3
    DEFAULT_RANDOM_UNIFORM_MAX = 1e-3


# ------------------------------------------------------------------------


class Initializer(type):
    """
    A metaclass for an initializer class
    """
    def __getattr__(cls, key):
        return cls[key]

    def __setattr__(cls, key, value):
        raise ValueError('Cannot set value to an initializer class.')

    @property
    def label(cls):
        """
        Get initializer label.
        Returns:
            str
        """
        return cls._label


# ------------------------------------------------------------------------


class Initializer(object, metaclass=Initializer):
    _label = INITIALIZER.LABEL
    """
    A base initializer class
    """

    def __str__(self):
        return self.label

    # ------------------------------------------------------------------------

    @property
    def label(self):
        """
        Get dropout label.
        Returns:
            str
        """
        return type(self).label

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return initializer state snapshot as a dict.
        Arguments:
            as_json: set to True to convert and return dict as JSON
            beautify_json: set to True to beautify JSON
        Returns:
            dict
        """
        snapshot = {
            'label': self.label,
            'base_label': Initializer.label
        }
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()

# ------------------------------------------------------------------------


class Zeros(Initializer):
    _label = INITIALIZER.ZEROS_LABEL
    """
    Initialize an array with shape with all zeros
    """
    def __init__(self):
        self._zeros_t = None
        super().__init__()

    @MType((int,), dtype=type(np.dtype))
    def __call__(self, shape, *, dtype=np.float32):
        if self._zeros_t is None or self._zeros_t.shape != shape:
            self._zeros_t = np.full(shape=shape, dtype=dtype, fill_value=0)
        return self._zeros_t.copy()

    # ------------------------------------------------------------------------

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return initializer state snapshot as a dict.
        Arguments:
            as_json: set to True to convert and return dict as JSON
            beautify_json: set to True to beautify JSON
        Returns:
            dict
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'dtype': str(self._zeros_t.dtype) if self._zeros_t is not None else None
        })
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()


# ------------------------------------------------------------------------


class Ones(Initializer):
    _label = INITIALIZER.ONES_LABEL
    """
    Initialize an array with shape with all ones
    """
    def __init__(self):
        self._ones_t = None
        super().__init__()

    @MType((int,), dtype=type(np.dtype))
    def __call__(self, shape, *, dtype=np.float32):
        if self._ones_t is None or self._ones_t.shape != shape:
            self._ones_t = np.full(shape=shape, dtype=dtype, fill_value=1)
        return self._ones_t.copy()

    # ------------------------------------------------------------------------

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return initializer state snapshot as a dict.
        Arguments:
            as_json: set to True to convert and return dict as JSON
            beautify_json: set to True to beautify JSON
        Returns:
            dict
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'dtype': str(self._ones_t.dtype) if self._ones_t is not None else None
        })
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()


# ------------------------------------------------------------------------


class Constant(Initializer):
    _label = INITIALIZER.CONSTANT_LABEL
    """
    Initialize an array with shape with a constant
    """
    @MType(OneOfType(int, float))
    def __init__(self, value):
        self._value = value
        self._const_t = None
        super().__init__()

    @MType((int,), dtype=type(np.dtype))
    def __call__(self, shape, *, dtype=np.float32):
        if self._const_t is None or self._const_t.shape != shape:
            self._const_t = np.full(shape=shape, dtype=dtype, fill_value=self._value)
        return self._const_t.copy()

    # ------------------------------------------------------------------------

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return initializer state snapshot as a dict.
        Arguments:
            as_json: set to True to convert and return dict as JSON
            beautify_json: set to True to beautify JSON
        Returns:
            dict
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'dtype': str(self._const_t.dtype) if self._const_t is not None else None,
            'value': self._value
        })
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()


# ------------------------------------------------------------------------


class Identity(Initializer):
    _label = INITIALIZER.IDENTITY_LABEL
    """
    Initialize an array with shape with an identity
    """
    def __init__(self):
        self._identity_m = None
        super().__init__()

    @MType((int,), dtype=type(np.dtype))
    def __call__(self, shape, *, dtype=np.float32):
        (row_size, col_size) = shape
        if row_size != col_size:
            raise ValueError('Identity initializer required square shape with rows = cols.')

        dim = row_size
        if self._identity_m is None or self._identity_m.shape != shape:
            self._identity_m = np.identity(n=dim, dtype=dtype)
        return self._identity_m.copy()

    # ------------------------------------------------------------------------

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return initializer state snapshot as a dict.
        Arguments:
            as_json: set to True to convert and return dict as JSON
            beautify_json: set to True to beautify JSON
        Returns:
            dict
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'dtype': str(self._identity_m.dtype) if self._identity_m is not None else None
        })
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()

# ------------------------------------------------------------------------


class Diagonal(Initializer):
    _label = INITIALIZER.DIAGONAL_LABEL
    """
    Initialize an array with shape with a diagonal
    """
    @MType(OneOfType(int, float))
    def __init__(self, value):
        self._value = value
        self._diagonal_t = None
        super().__init__()

    @MType((int,), dtype=type(np.dtype))
    def __call__(self, shape, *, dtype=np.float32):
        if self._diagonal_t is None or self._diagonal_t.shape != shape:
            self._diagonal_t = np.zeros(shape=shape, dtype=dtype)
            np.fill_diagonal(self._diagonal_t, self._value, wrap=False)
        return self._diagonal_t.copy()

    # ------------------------------------------------------------------------

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return initializer state snapshot as a dict.
        Arguments:
            as_json: set to True to convert and return dict as JSON
            beautify_json: set to True to beautify JSON
        Returns:
            dict
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'dtype': str(self._diagonal_t.dtype) if self._diagonal_t is not None else None,
            'value': self._value
        })
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()


# ------------------------------------------------------------------------


class RandomBinary(Initializer):
    _label = INITIALIZER.RANDOM_BINARY_LABEL
    """
    Initialize an array with shape with an random binary
    """
    @MType(seed=OneOfType(int, None))
    def __init__(self, *, seed=None):
        self._rbinary_t = None
        if seed is not None and seed < 0:
            warnings.warn(
                'Seed must be > 0. Reset to None',
                UserWarning)
            self._seed = None
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)
        super().__init__()

    @MType((int,), pzero=float, dtype=type(np.dtype), reuse=bool)
    def __call__(self, shape, *, pzero=0.5, dtype=np.int8, reuse=False):
            (row_size, col_size) = shape
            if pzero >= 1 or pzero <= 0:
                warnings.warn(
                    'Probability of zeros must be > 0 and < 1. Reset to 0.5.',
                    UserWarning)
                pzero = 0.5
            if self._rbinary_t is None or self._rbinary_t.shape != shape or not reuse:
                self._rbinary_t = self._rng.binomial(size=shape, n=1, p=1 - pzero)
            if self._rbinary_t.shape != shape and reuse:
                warnings.warn(
                    'Unable to reuse last random binary because the shape is different.',
                    UserWarning)
            return self._rbinary_t.copy()

    # ------------------------------------------------------------------------

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return initializer state snapshot as a dict.
        Arguments:
            as_json: set to True to convert and return dict as JSON
            beautify_json: set to True to beautify JSON
        Returns:
            dict
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'seed': self._seed,
            'dtype': str(self._rbinary_t.dtype) if self._rbinary_t is not None else None
            # 'values': self._rbinary_t.tolist() if self._rbinary_t is not None else None
        })
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()


# ------------------------------------------------------------------------


class RandomOrthonormal(Initializer):
    _label = INITIALIZER.RANDOM_ORTHONORMAL_LABEL
    """
    Initialize an array with shape with a random orthonormal
    """
    @MType(seed=OneOfType(int, None))
    def __init__(self, *, seed=None):
        self._rorthonormal_m = None
        if seed is not None and seed < 0:
            warnings.warn(
                'Seed must be > 0. Reset to None',
                UserWarning)
            self._seed = None
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)
        super().__init__()

    @MType((int,), dtype=type(np.dtype), reuse=bool)
    def __call__(self, shape, *, dtype=np.float32, reuse=False):
        (row_size, col_size) = shape
        if row_size != col_size:
            raise ValueError('RandomOrthonormal initializer requires shape to be square with rows = cols.')
        if self._rorthonormal_m is None or self._rorthonormal_m.shape != shape or not reuse:
            i_m = np.identity(n=row_size, dtype=dtype)
            one_v = np.ones(shape=(row_size,), dtype=dtype)
            for i in range(1, row_size):
                x_v = self._rng.normal(size=(row_size - i + 1,))
                one_v[i - 1] = np.sign(x_v[0])
                x_v -= one_v[i - 1] * np.sqrt((np.square(x_v)).sum())
                # householder transformation
                h_m = np.multiply(np.identity(n=(row_size - i + 1), dtype=dtype) - 2, np.outer(x_v, x_v)) / (np.square(x_v)).sum()
                mat = np.identity(n=row_size, dtype=dtype)
                mat[i - 1:, i - 1:] = h_m
                i_m = np.dot(i_m, mat)
                # fix the last sign such that the determinant is 1
            one_v[-1] = math.pow(-1, 1 - (row_size % 2)) * one_v.prod()

            # equivalent to np.dot(np.diag(one_v), i_m)
            i_m = np.multiply(one_v, i_m.transpose()).transpose()

            self._rorthonormal_m = i_m
        if self._rorthonormal_m.shape != shape and reuse:
            warnings.warn(
                'Unable to reuse last random orthonormal because the shape is different.',
                UserWarning)
        return self._rorthonormal_m.copy()

    # ------------------------------------------------------------------------

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return initializer state snapshot as a dict.
        Arguments:
            as_json: set to True to convert and return dict as JSON
            beautify_json: set to True to beautify JSON
        Returns:
            dict
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'seed': self._seed,
            'dtype': str(self._rorthonormal_m.dtype) if self._rorthonormal_m is not None else None
            # 'values': self._rorthonormal_m.tolist() if self._rorthonormal_m is not None else None
        })
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()


# ------------------------------------------------------------------------


class RandomNormal(Initializer):
    _label = INITIALIZER.RANDOM_NORMAL_LABEL
    """
    Initialize an array with shape with a random normal at a mean +/- variance
    """
    @MType(mean=float, variance=float, seed=OneOfType(int, None))
    def __init__(self, *,
                 mean=INITIALIZER.DEFAULT_RANDOM_NORMAL_MEAN,
                 variance=INITIALIZER.DEFAULT_RANDOM_NORMAL_VARIANCE,
                 seed=None):
        self._rnormal_t = None
        self._mean = mean
        self._variance = variance
        if seed is not None and seed < 0:
            warnings.warn(
                'Seed must be > 0. Reset to None',
                UserWarning)
            self._seed = None
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)
        super().__init__()

    @MType((int,), dtype=type(np.dtype), reuse=bool)
    def __call__(self, shape, *, dtype=np.float32, reuse=False):
        if self._rnormal_t is None or self._rnormal_t.shape != shape or not reuse:
            self._rnormal_t = self._rng.normal(loc=self._mean, scale=self._variance, size=shape).astype(dtype)
        if self._rnormal_t.shape != shape and reuse:
            warnings.warn(
                'Unable to reuse last random normal because the shape is different.',
                UserWarning)
        return self._rnormal_t.copy()

    # ------------------------------------------------------------------------

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return initializer state snapshot as a dict.
        Arguments:
            as_json: set to True to convert and return dict as JSON
            beautify_json: set to True to beautify JSON
        Returns:
            dict
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'seed': self._seed,
            'dtype': str(self._rnormal_t.dtype) if self._rnormal_t is not None else None,
            'mean': self._mean,
            'variance': self._variance
            # 'values': self._rnormal_t.tolist() if self._rnormal_t is not None else None
        })
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()


# ------------------------------------------------------------------------


class RandomUniform(Initializer):
    _label = INITIALIZER.RANDOM_UNIFORM_LABEL
    """
    Initialize an array with shape with a random uniform between min and max
    """
    @MType(min=float, max=float, seed=OneOfType(int, None))
    def __init__(self, *,
                 min=INITIALIZER.DEFAULT_RANDOM_UNIFORM_MIN,
                 max=INITIALIZER.DEFAULT_RANDOM_UNIFORM_MAX,
                 seed=None):
        self._runiform_t = None
        self._min = min
        self._max = max
        if seed is not None and seed < 0:
            warnings.warn(
                'Seed must be > 0. Reset to None',
                UserWarning)
            self._seed = None
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)
        super().__init__()

    @MType((int,), dtype=type(np.dtype), reuse=bool)
    def __call__(self, shape, *, dtype=np.float32, reuse=False):
        if self._runiform_t is None or self._runiform_t.shape != shape or not reuse:
            self._runiform_t = self._rng.uniform(low=self._min, high=self._max, size=shape).astype(dtype)
        if self._runiform_t.shape != shape and reuse:
            warnings.warn(
                'Unable to reuse last random uniform because the shape is different.',
                UserWarning)
        return self._runiform_t.copy()

    # ------------------------------------------------------------------------

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return initializer state snapshot as a dict.
        Arguments:
            as_json: set to True to convert and return dict as JSON
            beautify_json: set to True to beautify JSON
        Returns:
            dict
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'seed': self._seed,
            'dtype': str(self._runiform_t.dtype) if self._runiform_t is not None else None,
            'min': self._min,
            'max': self._max,
            'spread': (self._max - self._min) / 2
            # 'values': self._runiform_t.tolist() if self._runiform_t is not None else None
        })
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()


# ------------------------------------------------------------------------


class GlorotRandomNormal(Initializer):
    _label = INITIALIZER.GLOROT_RANDOM_NORMAL_LABEL
    """
    Initialize an array with shape with a glorot random normal at a 0 +/- sqrt(2 / sum(shape))
    """
    @MType(seed=OneOfType(int, None))
    def __init__(self, *, seed=None):
        self._grnormal_t = None
        if seed is not None and seed < 0:
            warnings.warn(
                'Seed must be > 0. Reset to None',
                UserWarning)
            self._seed = None
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)
        super().__init__()

    @MType((int,), dtype=type(np.dtype), reuse=bool)
    def __call__(self, shape, *, dtype=np.float32, reuse=False):
        if self._grnormal_t is None or self._grnormal_t.shape != shape or not reuse:
            variance = math.sqrt(2 / sum(shape))
            self._grnormal_t = self._rng.normal(loc=0, scale=variance, size=shape).astype(dtype)
        if self._grnormal_t.shape != shape and reuse:
            warnings.warn(
                'Unable to reuse last glorot normal because the shape is different.',
                UserWarning)
        return self._grnormal_t.copy()

    # ------------------------------------------------------------------------

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return initializer state snapshot as a dict.
        Arguments:
            as_json: set to True to convert and return dict as JSON
            beautify_json: set to True to beautify JSON
        Returns:
            dict
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'seed': self._seed,
        })
        if self._grnormal_t is not None:
            snapshot.update({
                'dtype': str(self._grnormal_t.dtype),
                'mean': 0,
                'variance': math.sqrt(2 / sum(self._grnormal_t.shape))
                # 'values': self._grnormal_t.tolist() if self._grnormal_t is not None else None
            })
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()

# ------------------------------------------------------------------------


class GlorotRandomUniform(Initializer):
    _label = INITIALIZER.GLOROT_RANDOM_UNIFORM_LABEL
    """
    Initialize an array with shape with a glorot random uniform at a 0 +/- sqrt(2 / sum(shape))
    """
    @MType(seed=OneOfType(int, None))
    def __init__(self, *, seed=None):
        self._gruniform_t = None
        if seed is not None and seed < 0:
            warnings.warn(
                'Seed must be > 0. Reset to None',
                UserWarning)
            self._seed = None
        self._seed = seed
        self._rng = np.random.RandomState(seed=self._seed)
        super().__init__()

    @MType((int,), dtype=type(np.dtype), reuse=bool)
    def __call__(self, shape, *, dtype=np.float32, reuse=False):
        if self._gruniform_t is None or self._gruniform_t.shape != shape or not reuse:
            spread = math.sqrt(2 / sum(shape)) / 2
            min = -spread
            max = spread
            self._gruniform_t = self._rng.uniform(low=min, high=max, size=shape).astype(dtype)
        if self._gruniform_t.shape != shape and reuse:
            warnings.warn(
                'Unable to reuse last glorot uniform because the shape is different.',
                UserWarning)
        return self._gruniform_t.copy()

    # ------------------------------------------------------------------------

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return initializer state snapshot as a dict.
        Arguments:
            as_json: set to True to convert and return dict as JSON
            beautify_json: set to True to beautify JSON
        Returns:
            dict
        """
        snapshot = super().snapshot(as_json=False, beautify_json=False)
        snapshot.update({
            'seed': self._seed
        })
        if self._gruniform_t is not None:
            spread = math.sqrt(2 / sum(self._gruniform_t.shape)) / 2
            min = -spread
            max = spread
            snapshot.update({
                'dtype': str(self._gruniform_t.dtype),
                'min': min,
                'max': max,
                'spread': spread
                # 'values': self._gruniform_t.tolist() if self._gruniform_t is not None else None
            })
        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()
