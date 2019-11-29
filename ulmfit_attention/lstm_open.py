from typing import *
from torch import Tensor
from torch import nn
import torch


class LSTMOpen(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers: int = 1, bias: bool = True,
                 batch_first: bool = False, dropout: float = 0., bidirectional: bool = False,
                 return_memory: bool = False):
        if bidirectional:
            raise NotImplementedError("Not supported yet")  # TODO: implement bidirectional
        if not batch_first:
            raise NotImplementedError("Not supported yet")  # TODO: implement batch_first=False
        if num_layers > 1:
            raise NotImplementedError("Not supported yet")  # TODO: implement multi-layer networks
        if dropout != 0.:
            raise NotImplementedError("Not supported yet")  # TODO: implement dropout as in LSTM
        super(LSTMOpen, self).__init__()
        self.return_memory = return_memory
        self.cell = nn.LSTMCell(input_size, hidden_size, bias)
        # enable LSTM-like access to parameters, so that load_state_dict() will work (with strict=False)
        self.weight_ih_l0 = self.cell.weight_ih
        self.weight_hh_l0 = self.cell.weight_hh
        self.bias_ih_l0 = self.cell.bias_ih
        self.bias_hh_l0 = self.cell.bias_hh

        # TODO: add option to log all memory states
        # TODO: solve loading/conversion from LSTM in a better way

    def forward(self, inp: Tensor, hidden: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        bs, sl, di = inp.shape
        hx, cx = (hidden[0].squeeze(0), hidden[1].squeeze(0))
        outputs = []
        for i in range(sl):
            hx, cx = self.cell(inp[:, i, :], (hx, cx))
            if self.return_memory:
                outputs.append(cx)
            else:
                outputs.append(hx)
        results = torch.stack(outputs).transpose(0, 1)
        if self.return_memory:
            results.tanh_()
        return results, (hx.unsqueeze(0), cx.unsqueeze(0))

    def load_params_from_lstm(self, orig: nn.LSTM):
        self.cell.weight_ih = orig.weight_ih_l0
        self.cell.weight_hh = orig.weight_hh_l0
        self.cell.bias_ih = orig.bias_ih_l0
        self.cell.bias_hh = orig.bias_hh_l0
