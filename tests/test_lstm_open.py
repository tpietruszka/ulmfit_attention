import torch
from torch import nn
from ulmfit_attention import lstm_open

bs = 80
sl = 120
in_sz = 100
out_sz = 200

input = torch.randn(bs, sl, in_sz)
h0 = torch.randn(1, bs, out_sz)
c0 = torch.randn(1, bs, out_sz)


def test_simple_case_lstm_equivalent():
    rnn = nn.LSTM(in_sz, out_sz, batch_first=True)
    rnn2 = lstm_open.LSTMOpen(in_sz, out_sz, batch_first=True)

    # copy the (random) parameters of the original LSTM, to enable comparison of results
    rnn2.cell.weight_ih = rnn.weight_ih_l0
    rnn2.cell.weight_hh = rnn.weight_hh_l0
    rnn2.cell.bias_ih = rnn.bias_ih_l0
    rnn2.cell.bias_hh = rnn.bias_hh_l0

    output2, (hn2, cn2) = rnn2(input, (h0, c0))
    assert output2.shape == (bs, sl, out_sz)
    assert hn2.shape == h0.shape
    assert cn2.shape == c0.shape

    output, (hn, cn) = rnn(input, (h0, c0))
    assert (output - output2).abs().sum() < 0.01  # will not be exact, but close enough
    assert (hn - hn2).abs().sum() < 0.01
    assert (cn - cn2).abs().sum() < 0.01


def test_load_from_lstm():
    rnn = nn.LSTM(in_sz, out_sz, batch_first=True)
    rnn2 = lstm_open.LSTMOpen(in_sz, out_sz, batch_first=True)
    rnn2.load_params_from_lstm(rnn)
    output, _ = rnn(input, (h0, c0))
    output2, _ = rnn2(input, (h0, c0))
    assert (output - output2).abs().sum() < 0.01  # will not be exact, but close enough
