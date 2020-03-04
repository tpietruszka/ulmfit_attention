from fastai.text import *
from fastai.text.learner import _model_meta
from hyperspace_explorer.configurables import ConfigurableDataclass, RegisteredAbstractMeta
from .classifier_head import SequenceAggregatingClassifier
from .aggregations import Aggregation


class Classifier(ConfigurableDataclass, metaclass=RegisteredAbstractMeta, is_registry=True):
    """
    Not really a classifier - these classes create Learner objects, which contain classifiers.
    """
    @abc.abstractmethod
    def get_learner(self, db: TextClasDataBunch) -> 'RNNLearner':
        pass


@dataclass
class BaselineClassifier(Classifier):
    drop_mult: float = 1.
    lin_ftrs: Collection[int] = field(default_factory=lambda: [50])

    def get_learner(self, db: TextClasDataBunch) -> 'RNNLearner':
        return text_classifier_learner(db, AWD_LSTM, lin_ftrs=self.lin_ftrs,
                                       drop_mult=self.drop_mult)


@dataclass
class AggregatingClassifier(Classifier):
    Aggregation: Dict
    drop_mult: float = 1.
    lin_ftrs: Collection[int] = field(default_factory=lambda: [50])

    def get_learner(self, db: TextClasDataBunch) -> 'RNNLearner':
        "Create a `Learner` with a text classifier from `db`"
        vocab_sz = len(db.vocab.itos)
        n_class = db.c
        model = self.get_text_classifier(vocab_sz, n_class)
        meta = self.get_encoder_meta()
        learn = RNNLearner(db, model, split_func=meta['split_clas'])
        return learn

    @staticmethod
    def get_encoder_arch():
        return AWD_LSTM

    def get_encoder_meta(self):
        """Fastai meta information about various encoders"""
        return _model_meta[self.get_encoder_arch()]

    def get_text_classifier(self, vocab_sz: int, n_class: int, bptt: int = 70, max_len: int = 20 * 70,
                            ps: Collection[float] = None, pad_idx: int = 1, rnn_layers_used=None) -> nn.Module:
        "Like Fastai's `get_text_classifier`, but with a choice of classifier head architecture"
        meta = self.get_encoder_meta()

        dv = meta['config_clas']['emb_sz']
        head_config = deepcopy(self.Aggregation)
        head_config['dv'] = dv
        pooling = Aggregation.from_config(head_config)

        config = meta['config_clas'].copy()
        for k in config.keys():
            if k.endswith('_p'): config[k] *= self.drop_mult
        if rnn_layers_used is None: rnn_layers_used = [-1]
        if ps is None:  ps = [0.1] * len(self.lin_ftrs)
        layers = self.lin_ftrs + [n_class]
        ps = [config.pop('output_p')] + ps
        init = config.pop('init') if 'init' in config else None
        enc_arch = self.get_encoder_arch()
        encoder = MultiBatchEncoder(bptt, max_len, enc_arch(vocab_sz, **config), pad_idx=pad_idx)
        head = SequenceAggregatingClassifier(pooling, layers, ps, rnn_layers_used)
        model = SequentialRNN(encoder, head)
        return model if init is None else model.apply(init)
