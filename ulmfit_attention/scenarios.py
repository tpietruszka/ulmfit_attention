import torch
import numpy as np
from fastai.text import Learner, DatasetType, accuracy
from typing import *
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


