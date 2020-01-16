import abc
from typing import *
from torch import nn, Tensor
from torch.functional import F
from fastai.text import bn_drop_lin
from .utils import RegisteredAbstractMeta, Configurable


class Aggregation(nn.Module, Configurable, metaclass=RegisteredAbstractMeta, is_registry=True):
    @abc.abstractmethod
    def forward(self, inp: Tensor, mask: Tensor = None) -> Tensor:
        pass

    @abc.abstractmethod
    def output_dim(self) -> int:
        """Should return the dimension of the returned hidden state (per-sample dimension)"""
        pass


class BranchingAttentionAggregation(Aggregation):
    def __init__(self, dv: int, att_hid_layers: Sequence[int], att_dropouts: Union[float, Sequence[float]],
                 agg_layers: Sequence[int], agg_dropouts: Union[float, Sequence[float]], att_bn: bool = False,
                 agg_bn: bool = False):
        super().__init__()
        att_layers = [dv] + list(att_hid_layers) + [1]
        self.head = MultiLayerPointwise(att_layers, att_dropouts, batchnorm=att_bn)
        self.agg_dims = agg_layers
        if agg_layers:
            self.agg = MultiLayerPointwise([dv] + list(agg_layers), agg_dropouts, batchnorm=agg_bn)
        self.dv = dv
        self.last_weights = None
        self.last_features = None

    @classmethod
    def get_default_config(cls) -> Dict:
        return {
            "att_hid_layers": [50],
            "att_dropouts": [0, 0],
            "att_bn": False,
            "agg_layers": [10],
            "agg_dropouts": 0,
            "agg_bn": False
        }

    @property
    def output_dim(self):
        return self.agg_dims[-1] if self.agg_dims else self.dv

    def forward(self, inp, mask=None):
        weights_unnorm = self.head(inp).squeeze(-1)
        if mask is not None:
            weights_unnorm = weights_unnorm.masked_fill_(mask, 0)
        weights = F.softmax(weights_unnorm, dim=1)
        self.last_weights = weights.detach().cpu()
        if self.agg_dims:
            to_agg = self.agg(inp)
        else:
            to_agg = inp
        self.last_features = to_agg.detach().cpu()
        weighted = to_agg * weights.unsqueeze(-1).expand_as(to_agg)
        res = weighted.sum(1)
        return res


class MultiLayerPointwise(nn.Module):
    """Applies a point-wise neural network, with ReLU activations between layers and none at the end"""

    def __init__(self, dims: Sequence[int], dropouts: Union[Sequence[float], float] = 0., batchnorm: bool = True):
        super().__init__()
        # TODO: activation function as parameter
        acts = [nn.ReLU(inplace=True)] * (len(dims) - 2) + [None]
        layers = []
        if not isinstance(dropouts, Sequence):
            dropouts = [dropouts] * (len(dims) - 1)
        for din, dout, act, drop in zip(dims[:-1], dims[1:], acts, dropouts):
            layers += bn_drop_lin(din, dout, bn=batchnorm, p=drop, actn=act)
        self.layers = nn.Sequential(*layers)

    def forward(self, inp):
        bs, sl, dv = inp.shape
        x = inp.view(bs * sl, -1)
        x = self.layers(x)
        x = x.view(bs, sl, -1)
        return x
