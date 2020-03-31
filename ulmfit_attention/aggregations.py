import abc
from typing import *
from fastai.text import bn_drop_lin
from hyperspace_explorer.configurables import RegisteredAbstractMeta, Configurable
from torch import nn, Tensor
from torch.functional import F


class Aggregation(nn.Module, Configurable, metaclass=RegisteredAbstractMeta, is_registry=True):
    @abc.abstractmethod
    def forward(self, inp: Tensor, mask: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def output_dim(self) -> int:
        """Should return the dimension of the returned hidden state (per-sample dimension)"""
        pass


class BranchingAttentionAggregation(Aggregation):
    def __init__(self, dv: int, att_hid_layers: Optional[Sequence[int]], att_dropouts: Union[float, Sequence[float]],
                 agg_layers: Sequence[int], agg_dropouts: Union[float, Sequence[float]], att_bn: bool, agg_bn: bool,
                 att_mask_fix: bool):
        super().__init__()
        if att_hid_layers is None:
            self.att = None
        else:
            att_layers = [dv] + list(att_hid_layers) + [1]
            self.att = MultiLayerPointwise(att_layers, att_dropouts, batchnorm=att_bn)
        self.agg_dims = agg_layers
        if agg_layers:
            self.agg = MultiLayerPointwise([dv] + list(agg_layers), agg_dropouts, batchnorm=agg_bn)
        self.dv = dv
        self.last_weights = None
        self.last_features = None
        if att_mask_fix:  # initially, by mistake, masking was done with 0s, before applying softmax
            self.pre_softmax_mask_fill = float("-inf")
        else:
            self.pre_softmax_mask_fill = 0

    @classmethod
    def get_default_config(cls) -> Dict:
        return {
            "att_hid_layers": [50],
            "att_dropouts": [0, 0],
            "att_bn": False,
            "att_mask_fix": False,  # recommended: True
            "agg_layers": [10],
            "agg_dropouts": 0,
            "agg_bn": False
        }

    @property
    def output_dim(self):
        return self.agg_dims[-1] if self.agg_dims else self.dv

    def forward(self, inp, mask):
        if self.att:
            weights_unnorm = self.att(inp).squeeze(-1)
            weights_unnorm = weights_unnorm.masked_fill_(mask, self.pre_softmax_mask_fill)
            weights = F.softmax(weights_unnorm, dim=1)
        else:
            weights_unnorm = mask.logical_not().type_as(inp)
            weights = weights_unnorm / weights_unnorm.sum(dim=1)[:, None]

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
