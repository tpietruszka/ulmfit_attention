from abc import ABCMeta, abstractmethod, ABC
from typing import *
from copy import deepcopy
import dataclasses

factories = {}


class RegisteredAbstractMeta(ABCMeta):
    """
    A class created by this metaclass and with `is_registry=True` will have a mapping of (name -> class) to all its
    subclasses in the `subclass_registry` class variable. Allows marking methods as abstract, as ABCMeta.

    Example:
    >>> class A(metaclass=RegisteredAbstractMeta, is_registry=True):
    ...     pass
    >>> class B(A):
    ...     def __init__(self, num):
    ...         self.num = num
    ...     def greet(self):
    ...         print(f'B-greet-{self.num}')
    >>> b_instance = A.factory('B', {'num':3})
    >>> b_instance.greet()
    B-greet-3
    """

    def __new__(mcs, name, bases, class_dct, **kwargs):
        x = super().__new__(mcs, name, bases, class_dct)
        if kwargs.get('is_registry', False):
            x.subclass_registry = {}
            x.factory = lambda cname, params: x.subclass_registry[cname](**params)
            if name in factories.keys():
                raise RuntimeError(f"Factory-class {name} defined more than once!")
            else:
                factories[name] = x
        else:
            x.subclass_registry[name] = x
        return x


class Configurable(ABC):
    """
    Base class for classes created with metaclass=RegisteredAbstractMeta
    Ensures that mappings of default values are provided in a consistent way and enables easy construction from
    a partial config
    """
    classNameField = 'className'

    @classmethod
    @abstractmethod
    def get_default_config(cls) -> Dict:
        pass

    @classmethod
    @abstractmethod
    def factory(cls, cname, params) -> 'Configurable':
        pass

    @classmethod
    def from_config(cls, params) -> 'Configurable':
        params = deepcopy(params)
        cname = params[cls.classNameField]
        del params[cls.classNameField]

        full = cls.subclass_registry[cname].get_default_config()
        full.update(params)
        return cls.factory(cname, full)


def dataclass_defaults_to_dict(cls: type) -> Dict:
    """Takes a dataclass-decorated class (not instance), returns a dict of default values"""
    res = {}
    for f in dataclasses.fields(cls):
        if not isinstance(f.default, dataclasses._MISSING_TYPE):
            res[f.name] = f.default
        if not isinstance(f.default_factory, dataclasses._MISSING_TYPE):
            res[f.name] = f.default_factory()
    return res
