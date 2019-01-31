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

from util.const import CONST
from util.validation import (
    MType,
    OneOfType
)

# ------------------------------------------------------------------------


class LAYER(CONST):
    LABEL = 'layer'
    ARRANGEMENT = ('', '')


class Layer(type):
    """
    A metaclass for layer base class.
    """
    def __getattr__(cls, key):
        return cls[key]

    def __setattr__(cls, key, value):
        raise ValueError('Cannot set value to a layer base class.')

    @property
    def label(cls):
        """
        Get layer label.
        Returns:
            str
        """
        return cls._label

    @property
    def arrangement(cls):
        """
        Get layer arrangement.
        Returns:
            tuple
        """
        return cls._arrangement


class Layer(object, metaclass=Layer):
    _label = LAYER.LABEL
    _arrangement = LAYER.ARRANGEMENT
    """
    Layer base class.
    Arguments:
        shape: layer shape
        name: layer name
    """
    @MType(shape=(int,), name=str)
    def __init__(self, *,
                 shape=(1, 1),
                 name=''):
        self._name = name
        self._next = None
        self._prev = None
        self._shape = None
        self._locked = False
        self.reconfig(shape=shape)

    def __str__(self):
        if self.name != '':
            return self.name + '_' + self.label
        else:
            return self.label

    def __iter__(self):
        """
        Set layer to be an iterator to allows iteration over all connected layers.
        """
        layer = self.head
        while layer is not None:
            yield layer
            layer = layer.next

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
    def arrangement(self):
        """
        Get layer arrangement.
        Returns:
            tuple
        """
        return type(self).arrangement

    @property
    def shape(self):
        """
        Get layer shape.
        Returns:
            tuple
        """
        return self._shape

    @property
    def size(self):
        """
        Get layer size.
        Returns:
            int
        """
        return self.shape[1]

    @property
    def index(self):
        """
        Get layer index.
        Returns:
            int
        """
        if self.has_prev:
            return self.prev.index + 1
        else:
            return 0

    @property
    def name(self):
        """
        Get layer name.
        Returns:
            str
        """
        return self._name

    @name.setter
    @MType(str)
    def name(self, name):
        """
        Set layer name.
        Arguments:
            name: string name
        """
        self._name = name

    @property
    def next(self):
        """
        Get next layer.
        Returns:
            layer
        """
        return self._next

    @property
    def prev(self):
        """
        Get previous layer.
        Returns:
            layer
        """
        return self._prev

    @property
    def head(self):
        """
        Get head layer.
        Returns:
            layer
        """
        if self.has_prev:
            if self.is_head:
                return self
            else:
                return self.prev.head
        else:
            return self

    @property
    def tail(self):
        """
        Get tail layer.
        Returns:
            layer
        """
        if self.has_next:
            if self.is_tail:
                return self
            else:
                return self.next.tail
        else:
            return self

    @property
    def has_prev(self):
        """
        Check if there is a connection to previous layer.
        Returns:
            bool
        """
        return self.prev is not None

    @property
    def has_next(self):
        """
        Check if there is a connection to next layer.
        Returns:
            bool
        """
        return self.next is not None

    @property
    def is_head(self):
        """
        Check if layer is head.
        Returns:
            bool
        """
        return self.is_singular or (self.next is not None and self.prev is None)

    @property
    def is_tail(self):
        """
        Check if layer is tail.
        Returns:
            bool
        """
        return self.is_singular or (self.next is None and self.prev is not None)

    @property
    def is_body(self):
        """
        Check if layer is center body.
        Returns:
            bool
        """
        return self.next is not None and self.prev is not None

    @property
    def is_singular(self):
        """
        Check if layer is a singular layer with no connection.
        Returns:
            bool
        """
        return self.next is None and self.prev is None

    @MType(Layer)
    def is_connected_to(self, layer):
        """
        Check if layer is already connected.
        Arguments:
            layer: layer to be check for connectivity
        Returns:
            bool
        """
        connected = False
        for connected_layer in self.head:
            connected = connected_layer is layer
            if connected:
                break
        return connected

    @property
    def is_locked(self):
        """
        Check if layer is locked.
        Returns:
            bool
        """
        return self._locked

    @property
    @abc.abstractmethod
    def inputs(self):
        """
        Get layer forward pass input. Not implemented.
        """
        pass

    @property
    @abc.abstractmethod
    def outputs(self):
        """
        Get layer forward pass output. Not implemented.
        """
        pass

    def lock(self):
        """
        Finalize by locking this layer and connecting layers in connection.
        """
        if not self.is_locked:
            self._locked = True
            if self.has_next:
                self.next.lock()
            if self.has_prev:
                self.prev.lock()

    def unlock(self):
        """
        Unlock this layer and connecting layers in connection.
        """
        if self.is_locked:
            self._locked = False
            if self.has_next:
                self.next.unlock()
            if self.has_prev:
                self.prev.unlock()

    @MType(shape=OneOfType((int,), None))
    def reconfig(self, *,
                 shape=None):
        """
        Reconfig layer.
        Arguments:
            shape:
        """
        if shape is not None:
            if not all(axis >= 1 for axis in shape):
                raise ValueError('Shape {shape} has axis < 1.'.format(shape=shape))
            if len(shape) < 2:
                raise ValueError('Shape must have atleast 2 axes.')
            if self.is_locked:
                warnings.warn('Layer {name} is locked. Reconfig layer shape skipped.'.format(name=self.name), UserWarning)
            if not self.is_singular:
                warnings.warn('Layer {name} has connection to other layers. Reconfig layer shape skipped.'.format(name=self.name), UserWarning)
            self._shape = shape
        self.reset()

    @MType(as_json=bool, beautify_json=bool)
    def snapshot(self, *, as_json=False, beautify_json=True):
        """
        Return layer state snapshot as a dict.
        Arguments:
            as_json: set to True to convert and return dict as JSON
            beautify_json: set to True to beautify JSON
        Returns:
            dict
        """
        snapshot = {
            'index': self.index,
            'name': self.name,
            'label': self.label,
            'base_label': Layer.label,
            'shape': self.shape,
            'locked': self.is_locked
        }

        if as_json:
            if beautify_json:
                return json.dumps(snapshot, indent=4, sort_keys=False)
            else:
                return json.dumps(snapshot)
        else:
            return snapshot.copy()

    @MType(int)
    def from_index(self, index):
        """
        Goto layer at index.
        Arguments:
            index: layer index
        Returns:
            layer
        """
        layer = self.head
        target_index = 0
        while layer is not None:
            if target_index == index:
                break
            if target_index > index:
                layer = None
                break
            if target_index < index:
                target_index += 1
            layer = layer.next
        if layer is None:
            warnings.warn(
                'No layer is found at index {index}.'.format(index=index),
                UserWarning)
        return layer

    @MType(Layer, position=str)
    def connect(self, layer, *, position='ahead'):
        """
        Add a new layer ahead or behind this layer.
        Arguments:
            layer: next layer to make connection to
            position: connection position, ahead or behind
        Returns:
            layer
        """
        if self.is_locked:
            warnings.warn('Cannot make connection from locked layer {name1} to layer {name2}. Connecting layer skipped.'.format(name1=self.name, name2=layer.name), UserWarning)
            return self
        elif layer.is_connected_to(self):
            warnings.warn(
                'Layer {name1} is already connected to {name2}.'.format(name1=layer.name, name2=self.name),
                UserWarning)
            return self
        else:
            if position == 'ahead':
                if not self.is_singular and (self.is_head or self.is_body):
                    if not layer.is_singular:
                        raise RuntimeError('Cannot make connection from layer {name1} to a non-singular layer {name2}. '.format(name1=self.name, name2=layer.name))
                    if layer.arrangement[0] not in self.arrangement[1]:
                        raise RuntimeError('Cannot make connection from layer {name1} to layer {name2}. Mismatched arrangement.'.format(name1=self.name, name2=layer.name))
                    if layer.arrangement[1] not in self._next.arrangement[0]:
                        raise RuntimeError('Cannot make connection from layer {name1} to layer {name2}. Mismatched arrangement.'.format(name1=layer.name, name2=self._next.name))
                    self._next._prev = layer
                    layer._next = self._next
                    layer._prev = self
                    self._next = layer
                    return layer
                elif self.is_singular or self.is_tail:
                    if layer.is_body or (layer.is_tail and layer.has_prev):
                        raise RuntimeError('Cannot make connection from layer {name1} to a non-signular layer {name2} that is either a body or tail. '.format(name1=self.name, name2=layer.name))
                    if layer.arrangement[0] not in self.arrangement[1]:
                        raise RuntimeError('Cannot make connection from layer {name1} to layer {name2}. Mismatched arrangement.'.format(name1=self.name, name2=layer.name))
                    self._next = layer
                    layer._prev = self
                    return layer
            elif position == 'behind':
                if not layer.is_singular:
                    raise RuntimeError('Cannot make connection from layer {name1} to a non-singular layer {name2}. '.format(name1=self.name, name2=layer.name))
                if self.is_singular or self.is_head:
                    if self.arrangement[0] not in layer.arrangement[1]:
                        raise RuntimeError('Cannot make connection from layer {name} to layer{name}. Mismatched arrangement.'.format(name1=layer.name, name2=self.name))
                    self._prev = layer
                    layer._next = self
                    return layer
                elif self.is_body or (not self.is_singular and self.is_tail):
                    if self.arrangement[0] not in layer.arrangement[1]:
                        raise RuntimeError('Cannot make connection from layer {name1} to layer {name2}. Mismatched arrangement.'.format(name1=layer.name, name2=self.name))
                    if layer.arrangement[0] not in self._prev.arrangement[1]:
                        raise RuntimeError('Cannot make connection from layer {name1} to layer {name2}. Mismatched arrangement.'.format(name1=self._prev.name, name2=layer.name))
                    self._prev._next = layer
                    layer._prev = self._prev
                    layer._next = self
                    self._prev = layer
                    return layer
            else:
                raise TypeError('Unknown position type {position}.'.format(position=position))

    @MType(Layer)
    def replace_with(self, layer):
        """
        Replace this layer with a different layer.
        Arguments:
            layer: layer to replace with
        Returns:
            layer
        """
        if self.is_locked:
            warnings.warn('Cannot replace locked layer {name1} with layer {name2}. Replace layer skipped.'.format(name1=self.name, name2=layer.name), UserWarning)
            return self
        elif layer.is_connected_to(self):
            warnings.warn(
                'Layer {name1} is already connected to {name2}.'.format(name1=layer.name, name2=self.name),
                UserWarning)
            return self
        else:
            if not layer.is_singular:
                raise RuntimeError('Cannot make connection from layer {name1} to non-singular layer {name2}. '.format(name1=self.name, name2=layer.name))
            if self.is_singular:
                raise RuntimeError('Cannot replace a non-connecting layer {name1} with layer {name2}.'.format(name1=self.name, name2=layer.name))
            if self.is_head:
                if self._next.arrangement[0] not in layer.arrangement[1]:
                    raise RuntimeError('Cannot make connection from layer {name1} to layer {name2}. Mismatched arrangement.'.format(name1=layer.name, name2=self._next.name))
                self._next._prev = layer
                layer._next = self._next
                self._next = None
            elif self.is_body:
                if self._next.arrangement[0] not in layer.arrangement[1]:
                    raise RuntimeError('Cannot make connection from layer {name1} to layer {name2}. Mismatched arrangement.'.format(name1=self._next.name, name2=layer.name))
                if layer.arrangement[0] not in self._prev.arrangement[1]:
                    raise RuntimeError('Cannot make connection from layer {name1} to layer {name2}. Mismatched arrangement.'.format(name1=layer.name, name2=self._prev.name))
                self._next._prev = layer
                self._prev._next = layer
                layer._next = self._next
                layer._prev = self._prev
                self._next = None
                self._prev = None
            elif self.is_tail:
                if layer.arrangement[0] not in self._prev.arrangement[1]:
                    raise RuntimeError('Cannot make connection from layer {name1} to layer {name2}. Mismatched arrangement.'.format(name1=self._prev.name, name2=layer.name))
                self._prev._next = layer
                layer._prev = self._prev
                self._prev = None
            return layer

    def disconnect(self):
        """
        Remove self from connection.
        Returns:
            layer
        """
        if self.is_locked:
            warnings.warn('Cannot remove locked layer {name}. Remove layer skipped.'.format(name=self.name), UserWarning)
            return self
        else:
            if self.is_head:
                self._next._prev = None
            elif self.is_body:
                if self._next.arrangement[0] not in self._prev.arrangement[1]:
                    raise RuntimeError('Cannot make connection from layer {name1} to layer {name2}. Mismatched arrangement.'.format(name1=self._prev.name, name2=self._next.name))
                self._next._prev = self._prev
                self._prev._next = self._next
            elif self.is_tail:
                self._prev._next = None
            else:
                raise RuntimeError('Cannot remove a non-connecting layer {name}. '.format(name=self.name))
            self._next = None
            self._prev = None
            return self

    @abc.abstractmethod
    def unassign_hooks(self):
        """
        Unassign all callback functions. Not implemented.
        """
        pass

    @abc.abstractmethod
    def assign_hook(self):
        """
        Assign callback functions. Not implemented.
        """
        pass

    @abc.abstractmethod
    def forward(self):
        """
        Layer forward pass method. Not implemented.
        """
        pass

    @abc.abstractmethod
    def backward(self):
        """
        Layer backward pass method. Not implemented.
        """
        pass
