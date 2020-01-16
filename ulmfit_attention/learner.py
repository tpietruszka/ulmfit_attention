from fastai.text import *
from fastai.text.learner import _model_meta
from .classifier_head import SequenceAggregatingClassifier
from .aggregations import Aggregation


def get_text_classifier_custom(enc_arch: Callable, head_config: Dict, vocab_sz: int, n_class: int,
                               bptt: int = 70, max_len: int = 20 * 70, config: dict = None, drop_mult: float = 1.,
                               lin_ftrs: Collection[int] = None, ps: Collection[float] = None,
                               pad_idx: int = 1, rnn_layers_used=None) -> nn.Module:
    "Like Fastai's `get_text_classifier`, but with a choice of classifier head architecture"
    meta = _model_meta[enc_arch]
    dv = meta['config_clas']['emb_sz']
    head_config = deepcopy(head_config)
    head_config['dv'] = dv
    pooling = Aggregation.from_config(head_config)

    config = ifnone(config, meta['config_clas']).copy()
    for k in config.keys():
        if k.endswith('_p'): config[k] *= drop_mult
    if rnn_layers_used is None: rnn_layers_used = [-1]
    if lin_ftrs is None: lin_ftrs = [50]
    if ps is None:  ps = [0.1] * len(lin_ftrs)
    layers = lin_ftrs + [n_class]
    ps = [config.pop('output_p')] + ps
    init = config.pop('init') if 'init' in config else None
    encoder = MultiBatchEncoder(bptt, max_len, enc_arch(vocab_sz, **config), pad_idx=pad_idx)
    head = SequenceAggregatingClassifier(pooling, layers, ps, rnn_layers_used)
    model = SequentialRNN(encoder, head)
    return model if init is None else model.apply(init)


def text_classifier_learner_custom(data: DataBunch, enc_arch: Callable, head_config: Dict,
                                   classifier_params: Dict = None, learner_params: Dict = None) -> 'RNNLearner':
    "Create a `Learner` with a text classifier from `data` and `arch`."
    learner_params = {} if learner_params is None else deepcopy(learner_params)
    classifier_params = {} if classifier_params is None else deepcopy(classifier_params)
    model = get_text_classifier_custom(enc_arch, head_config, len(data.vocab.itos), data.c,
                                       **classifier_params)
    meta = _model_meta[enc_arch]
    learn = RNNLearner(data, model, split_func=meta['split_clas'], **learner_params)
    return learn
