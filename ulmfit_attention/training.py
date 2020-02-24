from fastai.text import *
from hyperspace_explorer.configurables import RegisteredAbstractMeta, ConfigurableDataclass


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


class TrainingSchedule(ConfigurableDataclass, metaclass=RegisteredAbstractMeta, is_registry=True):
    @abc.abstractmethod
    def generate(self) -> List[Fit1CycleParams]:
        pass


@dataclass
class DefaultSchedule(TrainingSchedule):
    cycles_gradual: int = 1
    cycles_final: int = 2
    lr_init: float = 2e-2
    wd: float = 0.1

    def generate(self) -> List[Fit1CycleParams]:
        return [
            Fit1CycleParams(-1, self.cycles_gradual, self.lr_init, wd=self.wd),
            Fit1CycleParams(-2, self.cycles_gradual, self.lr_init / 2, wd=self.wd),
            Fit1CycleParams(-3, self.cycles_gradual, self.lr_init / 4, wd=self.wd),
            Fit1CycleParams(-5, self.cycles_final, self.lr_init / 20, wd=self.wd),
        ]


@dataclass
class HeadOnlySchedule(TrainingSchedule):
    cycles: int = 5
    lr: float = 2e-2
    wd: float = 0.1

    def generate(self) -> List[Fit1CycleParams]:
        return [
            Fit1CycleParams(-1, self.cycles, self.lr, wd=self.wd)
        ]
