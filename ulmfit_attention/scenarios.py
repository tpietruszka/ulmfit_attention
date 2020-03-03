import torch
import numpy as np
from fastai.text import Learner, DatasetType, accuracy
from typing import *
from hyperspace_explorer.scenario_base import Scenario
from ulmfit_attention import datasets
from ulmfit_attention import training
from ulmfit_attention.learner import Classifier


class SmallTrainSample(Scenario):
    @staticmethod
    def single_run(params) -> Tuple[float, Dict, Optional[Learner]]:
        dataset_params = params['Scenario']['Dataset']
        seed = params['seed']
        dataset = datasets.Dataset.from_config(dataset_params)
        data_bunch = dataset.get_training_sample(seed=seed)
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

        data_full = dataset.get_test_as_valid()
        learn.data = data_full
        pred, labels = learn.get_preds(DatasetType.Valid)

        # TODO: add options to include other metrics
        acc = float(accuracy(pred, labels))

        stats = {
            'train_losses': train_losses,
        }
        return acc, stats, learn
