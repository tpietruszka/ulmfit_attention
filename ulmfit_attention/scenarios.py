import torch
import numpy as np
from fastai.text import Learner, AWD_LSTM, DatasetType, accuracy
from typing import *
from hyperspace_explorer.scenario_base import Scenario
from ulmfit_attention import datasets
from ulmfit_attention import training
from ulmfit_attention.learner import text_classifier_learner_custom


class SmallTrainSample(Scenario):
    @staticmethod
    def single_run(params) -> Tuple[float, Dict, Optional[Learner]]:
        dataset_params = params['scenario']['dataset']
        seed = params['seed']
        dataset = datasets.Dataset.from_config(dataset_params)
        data_bunch = dataset.get_training_sample(seed=seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        learn = text_classifier_learner_custom(data_bunch, AWD_LSTM, params['aggregation'])
        _ = learn.load_encoder('fwd_enc')

        schedule = training.TrainingSchedule.from_config(params['training_schedule'])
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
