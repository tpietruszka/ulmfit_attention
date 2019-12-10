from abc import ABCMeta
from dataclasses import dataclass


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
        else:
            x.subclass_registry[name] = x
        return x


@dataclass
class Fit1CycleParams:
    """
    Describing one phase of fast.ai `fit_one_cycle` training.
    This class should be used like:
    ```
        params = Fit1CycleParams(-1, 1)
        learn.freeze_to(params.freeze_to)
        learn.fit_one_cycle(**params.to_dict())
    ```
    """
    freeze_to: int
    cyc_len: int
    lr_max_last: float = 1e-3
    lr_last_to_first_ratio: float = (2.6 ** 4)  # lr_max_first == lr_max_last / lr_last_to_first_ratio
    moms: (float, float) = (0.8, 0.7)
    div_factor: float = 25.0
    pct_start: float = 0.3
    wd: float = None

    @staticmethod
    def keys():
        return ['cyc_len', 'max_lr', 'moms', 'div_factor', 'pct_start', 'wd']

    def __getitem__(self, item):
        if item == 'max_lr':
            return slice(self.lr_max_last / self.lr_last_to_first_ratio, self.lr_max_last)
        return getattr(self, item)

    def to_dict(self):
        return {k: self[k] for k in self.keys()}