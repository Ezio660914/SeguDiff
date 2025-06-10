# -*- coding: utf-8 -*-
import os
import sys

import optuna
from optuna.trial import TrialState

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class IHyperTune:
    def __init__(self):
        self.study = None
        self.current_trial = None
        self.hyperparams = None

    def set_default_hyperparams(self, *args, **kwargs):
        raise NotImplementedError

    def suggest_hyperparams(self, *args, **kwargs):
        raise NotImplementedError

    def objective(self, trial: optuna.Trial):
        raise NotImplementedError

    def _tune_hyperparams(
            self,
            study_name=None,
            storage=None,
            load_if_exists=False,
            direction=None,
            n_trials=None,
    ):
        self.study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=load_if_exists, direction=direction)
        self.study.optimize(self.objective, n_trials=n_trials)

        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(self.study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = self.study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    def tune_hyperparams(self):
        raise NotImplementedError
