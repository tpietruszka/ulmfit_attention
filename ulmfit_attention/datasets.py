import abc
from typing import *
from fastai.text import *
from .utils import RegisteredAbstractMeta


class Dataset(metaclass=RegisteredAbstractMeta, is_registry=True):
    @abc.abstractmethod
    def get_training_sample(self, bs: int, size: int, seed: int) -> TextClasDataBunch:
        """If seed==0, should include the longest text"""
        pass

    @abc.abstractmethod
    def get_test_as_valid(self, bs: int) -> TextClasDataBunch:
        pass


class IMDB(Dataset):
    def __init__(self):
        super().__init__()
        self.path = untar_data(URLs.IMDB)
        self.vocab_path = self.path / 'itos.pkl'
        self.vocab = Vocab.load(self.vocab_path)
        self._test_set_cache = self.path / 'test_as_valid.pkl'

    def get_training_sample(self, bs: int, size: int, seed: int) -> TextClasDataBunch:
        default_processors = [OpenFileProcessor(), TokenizeProcessor(), NumericalizeProcessor(vocab=self.vocab)]
        return (TextList(self._sample_paths(size, seed, include_longest=(seed == 0)), vocab=self.vocab, path=self.path,
                         processor=default_processors)
                .split_none()
                .label_from_folder(classes=['neg', 'pos'])
                .databunch(bs=bs))

    def get_test_as_valid(self, bs: int) -> TextClasDataBunch:
        try:
            ds = load_data(self.path, self._test_set_cache, bs=bs)
        except FileNotFoundError:
            print('Loading the test set from source, no cached version found')
            ds = (TextList.from_folder(self.path, vocab=self.vocab)
                  .split_by_folder(valid='test')
                  .label_from_folder(classes=['neg', 'pos'])
                  .databunch(bs=bs))
            ds.save(self._test_set_cache)
        return ds

    def _sample_paths(self, size: int, seed: int, include_longest: bool = False) -> List[Path]:
        "Balanced sample of the IMDB train set. Include_longest can be used early on to ensure OOM wont happen later"
        s = random.getstate()
        random.seed(seed)
        if include_longest:
            # including biggest files from each class. Not guaranteed to have the most tokens, but should be good enough
            pos_all, neg_all = [
                sorted(list((self.path / 'train' / label).iterdir()), key=lambda x: x.stat().st_size, reverse=True)
                for label in ['pos', 'neg']]
            pos, neg = [[lst[0]] + random.sample(lst, size // 2 - 1) for lst in [pos_all, neg_all]]
        else:
            pos, neg = [random.sample(list((self.path / 'train' / label).iterdir()), size // 2)
                        for label in ['pos', 'neg']]
        random.setstate(s)
        return pos + neg

