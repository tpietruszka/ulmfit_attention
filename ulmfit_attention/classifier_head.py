from fastai.text import *
from torch import nn
from typing import *


class SequenceAggregatingClassifier(nn.Module):
    def __init__(self, agg_mod: nn.Module, layers: Collection[int], drops: Collection[float],
                 output_layers: List[int]):
        super().__init__()
        self.attn = agg_mod
        layers = [self.attn.output_dim] + list(layers)
        mod_layers = []
        activs = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]
        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):
            mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*mod_layers)
        self.output_layers = output_layers

    def forward(self, input: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        raw_outputs, outputs, mask = input
        if len(self.output_layers) == 1:
            output = outputs[self.output_layers[0]]
        else:
            output = torch.cat([outputs[x] for x in self.output_layers], 2)
        bs, sl, _ = output.size()  # bs, sl, nh
        x = self.attn(output, mask)
        x = self.layers(x)
        return x, raw_outputs, outputs
