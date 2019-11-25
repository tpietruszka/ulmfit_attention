from typing import *
from torch import Tensor
from torch import nn
import torch


class LSTMOpen(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers: int = 1, bias: bool = True,
                 batch_first: bool = False, dropout: float = 0., bidirectional: bool = False):
        if bidirectional:
            raise NotImplementedError("Not supported yet")  # TODO: implement bidirectional
        if not batch_first:
            raise NotImplementedError("Not supported yet")  # TODO: implement batch_first=False
        if num_layers > 1:
            raise NotImplementedError("Not supported yet")  # TODO: implement multi-layer networks
        if dropout != 0.:
            raise NotImplementedError("Not supported yet")  # TODO: implement dropout as in LSTM
        super(LSTMOpen, self).__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size, bias)

        # TODO: write a test
        # TODO: implement reset
        # TODO: implement loading
        # TODO: add logging all memory states

    def forward(self, input: Tensor, hidden: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        bs, sl, di = input.shape
        new_hidden = hidden
        outputs = []
        for i in range(sl):
            out, new_hidden = self.cell(input[:, i, :], new_hidden)
            print(f'{i}: {out.shape} - {new_hidden.shape}')
            outputs.append(out)
        return torch.cat(outputs), new_hidden
