import copy
import gc
import torch
import numpy as np
from typing import *
from fastai.text import Learner, DatasetType, accuracy
from hyperspace_explorer.scenario_base import Scenario
from ulmfit_attention import datasets
from ulmfit_attention import training
from ulmfit_attention.learner import Classifier


class SmallTrainSample(Scenario):
    def __init__(self, Dataset: Dict):
        super().__init__()
        self.dataset = datasets.Dataset.from_config(Dataset)

    def single_run(self, params) -> Tuple[float, Learner]:
        seed = params['seed']
        data_bunch = self.dataset.get_training_sample(seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        c = Classifier.from_config(params['Classifier'])
        learn = c.get_learner(data_bunch)
        _ = learn.load_encoder('fwd_enc')

        schedule = training.TrainingSchedule.from_config(params['TrainingSchedule'])
        train_losses = []

        for phase in schedule.generate():
            learn.freeze_to(phase.freeze_to)
            learn.fit_one_cycle(**phase.to_dict())
            train_losses.append([float(x) for x in learn.recorder.losses])

        data_full = self.dataset.get_test_as_valid()
        learn.data = data_full
        pred, labels = learn.get_preds(DatasetType.Valid)

        # TODO: add options to include other metrics
        acc = float(accuracy(pred, labels))

        self.info['train_losses'] = train_losses
        return acc, learn


class RepeatedSmallTrainSample(Scenario):
    def __init__(self, Dataset: Dict, num_folds: int):
        super().__init__()
        self.Dataset = Dataset
        self.num_folds = num_folds

    def single_run(self, params) -> Tuple[float, None]:
        params = copy.deepcopy(params)
        seed = params['seed']

        accuracies = []
        for i in range(seed, seed + self.num_folds):
            fold = SmallTrainSample(self.Dataset)
            params['seed'] = i
            acc, _ = fold.single_run(params)
            del _
            gc.collect()
            accuracies.append(acc)
            self.log_scalar('fold_accuracy', acc, i)

            self.info[i] = {}
            if fold._metrics:
                self.info[i]['metrics'] = fold._metrics
            if fold.info:
                self.info[i].update(fold.info)

        mean_accuracy = sum(accuracies) / len(accuracies)
        return mean_accuracy, None
