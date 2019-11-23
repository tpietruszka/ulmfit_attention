from abc import ABCMeta


class RegisteredAbstractMeta(ABCMeta):
    """
    A class created by this metaclass and with `is_registry=True` will have a mapping of (name -> class) to all its
    subclasses in the `subclass_registry` class variable. Allows marking methods as abstract, as ABCMeta.

    Example:
    >>> class A(metaclass=RegisteredAbstractMeta, is_registry=True):
    ...     pass
    >>> class B(A):
    ...     def greet(self):
    ...         print('B-greet')
    >>> b_instance = A.subclass_registry['B']()
    >>> b_instance.greet()
    B-greet
    """

    def __new__(mcs, name, bases, class_dct, **kwargs):
        x = super().__new__(mcs, name, bases, class_dct)
        if kwargs.get('is_registry', False):
            x.subclass_registry = {}
        else:
            x.subclass_registry[name] = x
        return x
