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

# ------------------------------------------------------------------------

DISABLE_VALIDATION = False

# ------------------------------------------------------------------------


class Validator(type):
    """
    A metaclass for validator class.
    """
    def __getattr__(cls, key):
        return cls[key]

    def __setattr__(cls, key, value):
        raise ValueError('Cannot set value to a validator class.')


class OneOfType(object, metaclass=Validator):
    """
    Checker if arg is one of type in list.
    Arguments:
        args: a list of argument types, [int, str, bool, dict, ... ]
    """
    def __init__(self, *args):
        if args:
            self._arg_types = args
        else:
            raise ValueError('OneOfType argument type list is empty.')

    @property
    def arg_types(self):
        """
        Get argument type list.
        Returns:
            list
        """
        return self._arg_types

    def is_matched(self, arg):
        """
        Check if argument matches with type in argument type list.
        Arguments:
            arg: an argument
        Returns:
            bool
        """
        matched = False
        for arg_type in self._arg_types:
            if arg_type is None:
                matched = arg is None
            elif arg_type == callable:
                matched = callable(arg)
            elif not isinstance(arg_type, tuple) and not isinstance(arg_type, list):
                if isinstance(arg_type, type) and isinstance(type(arg), type) and not issubclass(type(arg), arg_type):
                    matched = isinstance(type(arg), arg_type)
                else:
                    matched = isinstance(arg, arg_type)
            else:
                if isinstance(arg_type, tuple) and isinstance(arg, tuple):
                    inner_matched = True
                    for inner_arg in arg:
                        for inner_arg_type in arg_type:
                            if isinstance(inner_arg_type, type) and isinstance(type(inner_arg), type) and not issubclass(type(inner_arg), inner_arg_type):
                                inner_matched = isinstance(type(inner_arg), inner_arg_type)
                            elif inner_arg_type == callable:
                                inner_matched = callable(inner_arg)
                            else:
                                inner_matched = isinstance(inner_arg, inner_arg_type)
                            if not inner_matched:
                                break
                        if not inner_matched:
                            break
                    matched = inner_matched
                elif isinstance(arg_type, list) and isinstance(arg, list):
                    inner_matched = False
                    for inner_arg in arg:
                        for inner_arg_type in arg_type:
                            if isinstance(inner_arg_type, type) and isinstance(type(inner_arg), type) and not issubclass(type(inner_arg), inner_arg_type):
                                inner_matched = isinstance(type(inner_arg), inner_arg_type)
                            elif inner_arg_type == callable:
                                inner_matched = callable(inner_arg)
                            elif inner_arg_type is None:
                                inner_matched = inner_arg == inner_arg_type
                            else:
                                inner_matched = isinstance(inner_arg, inner_arg_type)
                    matched = inner_matched
            if matched:
                break
        return matched


class FType(object, metaclass=Validator):
    """
    Validate argument types for functions.
    Arguments:
        args:
        kwarg:
    """

    def __init__(self, *args, **kwargs):
        self._arg_types = args
        self._kwarg_types = kwargs

    def __call__(self, decorated_method):
        """
        Arguments:
            decorated_method:
            Returns:
                callable
        """
        def decorator(*args, **kwargs):
            """
            Arguments:
            Returns:
                callable
            """
            if DISABLE_VALIDATION:
                return decorated_method(*args, **kwargs)
            args_len = len(args)
            kwargs_len = len(kwargs)
            arg_types_len = len(self._arg_types)
            kwarg_types_len = len(self._kwarg_types)

            if (arg_types_len + kwarg_types_len) < (args_len + kwargs_len):
                raise ValueError(
                    'Mismatched number of arguments and argument validation types. Expecting {len1} or less, but recevied {len2}.'.format(
                        len1=arg_types_len,
                        len2=args_len + kwargs_len
                    )
                )
            if args_len > 0:
                for (arg, arg_type) in zip(args, self._arg_types):
                    if isinstance(arg_type, tuple):
                        if not isinstance(arg, tuple):
                            raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=tuple, type2=arg))
                        for inner_arg in arg:
                            for inner_arg_type in arg_type:
                                if isinstance(inner_arg_type, OneOfType):
                                    oneOfType = inner_arg_type
                                    if not oneOfType.is_matched(inner_arg):
                                        raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=oneOfType.arg_types, type2=type(inner_arg)))
                                else:
                                    if isinstance(inner_arg_type, type) and isinstance(type(inner_arg), type) and not issubclass(type(inner_arg), inner_arg_type):
                                        if not isinstance(type(inner_arg), inner_arg_type):
                                            raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=inner_arg_type, type2=type(inner_arg)))
                                    elif inner_arg_type == callable:
                                        if not callable(inner_arg):
                                            raise TypeError('Mismatched argument type. Expecting function or method, but recevied {type}.'.format(type=type(inner_arg)))
                                    elif not isinstance(inner_arg, inner_arg_type):
                                        raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=inner_arg_type, type2=inner_arg))
                    elif isinstance(arg_type, list):
                        if not isinstance(arg, list):
                            raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=list, type2=arg))
                        for inner_arg in arg:
                            for inner_arg_type in arg_type:
                                if isinstance(inner_arg_type, OneOfType):
                                    oneOfType = inner_arg_type
                                    if not oneOfType.is_matched(inner_arg):
                                        raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=oneOfType.arg_types, type2=type(inner_arg)))
                                else:
                                    if isinstance(inner_arg_type, type) and isinstance(type(inner_arg), type) and not issubclass(type(inner_arg), inner_arg_type):
                                        if not isinstance(type(inner_arg), inner_arg_type):
                                            raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=inner_arg_type, type2=type(inner_arg)))
                                    elif inner_arg_type == callable:
                                        if not callable(inner_arg):
                                            raise TypeError('Mismatched argument type. Expecting function or method, but recevied {type}.'.format(type(type=inner_arg)))
                                    elif not isinstance(inner_arg, inner_arg_type):
                                        raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=inner_arg_type, type2=inner_arg))
                    elif isinstance(arg_type, type) and isinstance(type(arg), type) and not issubclass(type(arg), arg_type):
                        if not isinstance(type(arg), arg_type):
                            raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=arg_type, type2=type(arg)))
                    elif isinstance(arg_type, OneOfType):
                        oneOfType = arg_type
                        if not oneOfType.is_matched(arg):
                            raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=oneOfType.arg_types, type2=type(arg)))
                    elif arg_type == callable:
                        if not callable(arg):
                            raise TypeError('Mismatched argument type. Expecting function or method, but recevied {type}.'.format(type(type=arg)))
                    else:
                        if not isinstance(arg, arg_type):
                            raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=arg_type, type2=type(arg)))
            if kwargs_len > 0:
                for (kwarg_key, kwarg) in kwargs.items():
                    if kwarg_key in self._kwarg_types:
                        kwarg_type = self._kwarg_types[kwarg_key]
                        if isinstance(kwarg_type, tuple):
                            if not isinstance(kwarg, tuple):
                                raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=tuple, type2=kwarg))
                            for inner_kwarg in kwarg:
                                for inner_kwarg_type in kwarg_type:
                                    if isinstance(inner_kwarg_type, OneOfType):
                                        oneOfType = inner_kwarg_type
                                        if not oneOfType.is_matched(inner_kwarg):
                                            raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=oneOfType.arg_types, type2=type(inner_kwarg)))
                                    else:
                                        if isinstance(inner_kwarg_type, type) and isinstance(type(inner_kwarg), type) and not issubclass(type(inner_kwarg), inner_kwarg_type):
                                            if not isinstance(type(inner_kwarg), inner_kwarg_type):
                                                raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=inner_kwarg, type2=type(inner_kwarg)))
                                        elif inner_kwarg_type == callable:
                                            if not callable(inner_kwarg):
                                                raise TypeError('Mismatched argument type. Expecting function or method, but recevied {type}.'.format(type(type=inner_kwarg)))
                                        elif not isinstance(inner_kwarg, inner_kwarg_type):
                                            raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=inner_kwarg_type, type2=inner_kwarg))
                        elif isinstance(kwarg_type, list):
                            if not isinstance(kwarg, list):
                                raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=list, type2=kwarg))
                            for inner_kwarg in kwarg:
                                for inner_kwarg_type in kwarg_type:
                                    if isinstance(inner_kwarg_type, OneOfType):
                                        oneOfType = inner_kwarg_type
                                        if not oneOfType.is_matched(inner_kwarg):
                                            raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=oneOfType.arg_types, type2=type(inner_kwarg)))
                                    else:
                                        if isinstance(inner_kwarg_type, type) and isinstance(type(inner_kwarg), type) and not issubclass(type(inner_kwarg), inner_kwarg_type):
                                            if not isinstance(type(inner_kwarg), inner_kwarg_type):
                                                raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=inner_kwarg, type2=type(inner_kwarg)))
                                        elif inner_kwarg_type == callable:
                                            if not callable(inner_kwarg):
                                                raise TypeError('Mismatched argument type. Expecting function or method, but recevied {type}.'.format(type(type=inner_kwarg)))
                                        elif not isinstance(inner_kwarg, inner_kwarg_type):
                                            raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=inner_kwarg_type, type2=inner_kwarg))
                        elif isinstance(kwarg_type, type) and isinstance(type(kwarg), type) and not issubclass(type(kwarg), kwarg_type):
                            if not isinstance(type(kwarg), kwarg_type):
                                raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=kwarg_type, type2=type(kwarg)))
                        elif isinstance(kwarg_type, OneOfType):
                            oneOfType = kwarg_type
                            if not oneOfType.is_matched(kwarg):
                                raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=oneOfType.arg_types, type2=type(kwarg)))
                        elif kwarg_type == callable:
                            if not callable(kwarg):
                                raise TypeError('Mismatched argument type. Expecting function or method, but recevied {type}.'.format(type(kwarg)))
                        else:
                            if not isinstance(kwarg, kwarg_type):
                                raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=kwarg_type, type2=type(kwarg)))

            return decorated_method(*args, **kwargs)
        return decorator


class MType(object, metaclass=Validator):
    """
    Validate input argument types for methods.
    Arguments:
        args:
        kwarg:
    """

    def __init__(self, *args, **kwargs):
        self._arg_types = args
        self._kwarg_types = kwargs

    def __call__(self, decorated_method):
        """
        Arguments:
        """
        def decorator(*args, **kwargs):
            """
            Arguments:
            Returns:
                callable
            """
            if DISABLE_VALIDATION:
                return decorated_method(*args, **kwargs)
            arg_self = args[0]
            args = args[1:]
            args_len = len(args)
            kwargs_len = len(kwargs)
            arg_types_len = len(self._arg_types)
            kwarg_types_len = len(self._kwarg_types)

            if (arg_types_len + kwarg_types_len) < (args_len + kwargs_len):
                raise TypeError(
                    'Mismatched number of arguments and argument validation types. Expecting {len1} or less, but recevied {len2}.'.format(
                        len1=arg_types_len,
                        len2=args_len + kwargs_len
                    )
                )
            if args_len > 0:
                for (arg, arg_type) in zip(args, self._arg_types):
                    if isinstance(arg_type, tuple):
                        if not isinstance(arg, tuple):
                            raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=tuple, type2=arg))
                        for inner_arg in arg:
                            for inner_arg_type in arg_type:
                                if isinstance(inner_arg_type, OneOfType):
                                    oneOfType = inner_arg_type
                                    if not oneOfType.is_matched(inner_arg):
                                        raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=oneOfType.arg_types, type2=type(inner_arg)))
                                else:
                                    if isinstance(inner_arg_type, type) and isinstance(type(inner_arg), type) and not issubclass(type(inner_arg), inner_arg_type):
                                        if not isinstance(type(inner_arg), inner_arg_type):
                                            raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=inner_arg_type, type2=type(inner_arg)))
                                    elif inner_arg_type == callable:
                                        if not callable(inner_arg):
                                            raise TypeError('Mismatched argument type. Expecting function or method, but recevied {type}.'.format(type=type(inner_arg)))
                                    elif not isinstance(inner_arg, inner_arg_type):
                                        raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=inner_arg_type, type2=inner_arg))
                    elif isinstance(arg_type, list):
                        if not isinstance(arg, list):
                            raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=list, type2=arg))
                        for inner_arg in arg:
                            for inner_arg_type in arg_type:
                                if isinstance(inner_arg_type, OneOfType):
                                    oneOfType = inner_arg_type
                                    if not oneOfType.is_matched(inner_arg):
                                        raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=oneOfType.arg_types, type2=type(inner_arg)))
                                else:
                                    if isinstance(inner_arg_type, type) and isinstance(type(inner_arg), type) and not issubclass(type(inner_arg), inner_arg_type):
                                        if not isinstance(type(inner_arg), inner_arg_type):
                                            raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=inner_arg_type, type2=type(inner_arg)))
                                    elif inner_arg_type == callable:
                                        if not callable(inner_arg):
                                            raise TypeError('Mismatched argument type. Expecting function or method, but recevied {type}.'.format(type=type(inner_arg)))
                                    elif not isinstance(inner_arg, inner_arg_type):
                                        raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=inner_arg_type, type2=inner_arg))
                    elif isinstance(arg_type, type) and isinstance(type(arg), type) and not issubclass(type(arg), arg_type):
                        if not isinstance(type(arg), arg_type):
                            raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=arg_type, type2=type(arg)))
                    elif isinstance(arg_type, OneOfType):
                        oneOfType = arg_type
                        if not oneOfType.is_matched(arg):
                            raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=oneOfType.arg_types, type2=type(arg)))
                    elif arg_type == callable:
                        if not callable(arg):
                            raise TypeError('Mismatched argument type. Expecting function or method, but recevied {type}.'.format(type=type(arg)))
                    else:
                        if not isinstance(arg, arg_type):
                            raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=arg_type, type2=type(arg)))
            if kwargs_len > 0:
                for (kwarg_key, kwarg) in kwargs.items():
                    if kwarg_key in self._kwarg_types:
                        kwarg_type = self._kwarg_types[kwarg_key]
                        if isinstance(kwarg_type, tuple):
                            if not isinstance(kwarg, tuple):
                                raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=tuple, type2=kwarg))
                            for inner_kwarg in kwarg:
                                for inner_kwarg_type in kwarg_type:
                                    if isinstance(inner_kwarg_type, OneOfType):
                                        oneOfType = inner_kwarg_type
                                        if not oneOfType.is_matched(inner_kwarg):
                                            raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=oneOfType.arg_types, type2=type(inner_kwarg)))
                                    else:
                                        if isinstance(inner_kwarg_type, type) and isinstance(type(inner_kwarg), type) and not issubclass(type(inner_kwarg), inner_kwarg_type):
                                            if not isinstance(type(inner_kwarg), inner_kwarg_type):
                                                raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=inner_kwarg, type2=type(inner_kwarg)))
                                        elif inner_kwarg_type == callable:
                                            if not callable(inner_kwarg):
                                                raise TypeError('Mismatched argument type. Expecting function or method, but recevied {type}.'.format(type(type=inner_kwarg)))
                                        elif not isinstance(inner_kwarg, inner_kwarg_type):
                                            raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=inner_kwarg_type, type2=inner_kwarg))
                        elif isinstance(kwarg_type, list):
                            if not isinstance(kwarg, list):
                                raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=list, type2=kwarg))
                            for inner_kwarg in kwarg:
                                for inner_kwarg_type in kwarg_type:
                                    if isinstance(inner_kwarg_type, OneOfType):
                                        oneOfType = inner_kwarg_type
                                        if not oneOfType.is_matched(inner_kwarg):
                                            raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=oneOfType.arg_types, type2=type(inner_kwarg)))
                                    else:
                                        if isinstance(inner_kwarg_type, type) and isinstance(type(inner_kwarg), type) and not issubclass(type(inner_kwarg), inner_kwarg_type):
                                            if not isinstance(type(inner_kwarg), inner_kwarg_type):
                                                raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=inner_kwarg, type2=type(inner_kwarg)))
                                        elif inner_kwarg_type == callable:
                                            if not callable(inner_kwarg):
                                                raise TypeError('Mismatched argument type. Expecting function or method, but recevied {type}.'.format(type(type=inner_kwarg)))
                                        elif not isinstance(inner_kwarg, inner_kwarg_type):
                                            raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=inner_kwarg_type, type2=inner_kwarg))
                        elif isinstance(kwarg_type, type) and isinstance(type(kwarg), type) and not issubclass(type(kwarg), kwarg_type):
                            if not isinstance(type(kwarg), kwarg_type):
                                raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=kwarg_type, type2=type(kwarg)))
                        elif isinstance(kwarg_type, OneOfType):
                            oneOfType = kwarg_type
                            if not oneOfType.is_matched(kwarg):
                                raise TypeError('Mismatched argument type. Expecting one of {type1}, but recevied {type2}.'.format(type1=oneOfType.arg_types, type2=type(kwarg)))
                        elif kwarg_type == callable:
                            if not callable(kwarg):
                                raise TypeError('Mismatched argument type. Expecting function or method, but recevied {type}.'.format(type=type(kwarg)))
                        else:
                            if not isinstance(kwarg, kwarg_type):
                                raise TypeError('Mismatched argument type. Expecting {type1}, but recevied {type2}.'.format(type1=kwarg_type, type2=type(kwarg)))

            return decorated_method(*(arg_self, *args), **kwargs)
        return decorator


class MShape(object, metaclass=Validator):
    """
    Validate input tensor shape.
    Arguments:
        axis: the axis to check the shape
        transpose: check the transpose of shape
    """
    @MType(axis=int, transpose=bool)
    def __init__(self, *, axis=2, transpose=False):
        if axis < -1:
            raise ValueError('Invalid axis {axis}'.format(axis=axis))
        else:
            self._axis = axis
            self._transpose = transpose

    def __call__(self, decorated_method):
        """
        Arguments:
        """
        def decorator(*args, **kwargs):
            """
            Arguments:
            Returns:
                callable
            """
            if DISABLE_VALIDATION:
                return decorated_method(*args, **kwargs)
            arg_self = args[0]
            args = args[1:]
            ref_shape = getattr(arg_self, 'shape', None)
            if ref_shape is None:
                raise TypeError('Missing shape attribute in {arg_self}'.format(arg_self=arg_self))
            else:
                if self._transpose:
                    ref_shape = tuple(reversed(ref_shape))
                for arg in args:
                    shape = getattr(arg, 'shape', None)
                    if shape is not None:
                        if len(shape) > 2:
                            shape = (1,) + shape[1:]
                        if self._axis == -1:
                            if ref_shape != shape:
                                raise TypeError('Mismatched tensor shape. Expecting {shape1}, but recevied {shape2}.'.format(shape1=ref_shape, shape2=shape))
                        else:
                            if ref_shape[self._axis] != shape[self._axis]:
                                raise TypeError('Mismatched tensor shape. Expecting {shape1}, but recevied {shape2}.'.format(shape1=ref_shape, shape2=shape))
                for kwarg in kwargs.values():
                    shape = getattr(kwarg, 'shape', None)
                    if shape is not None:
                        if len(shape) > 2:
                            shape = (1,) + shape[1:]
                        if self._axis == 2:
                            if ref_shape != shape:
                                raise TypeError('Mismatched tensor shape. Expecting {shape1}, but recevied {shape2}.'.format(shape1=ref_shape, shape2=shape))
                        elif self._axis == 1:
                            if ref_shape[1] != shape[1]:
                                raise TypeError('Mismatched tensor column shape. Expecting {shape1}, but recevied {shape2}.'.format(shape1=ref_shape, shape2=shape))
                        elif self._axis == 0:
                            if ref_shape[0] != shape[0]:
                                raise TypeError('Mismatched tensor row shape. Expecting {shape1}, but recevied {shape2}.'.format(shape1=ref_shape, shape2=shape))
                return decorated_method(*(arg_self, *args), **kwargs)
        return decorator
