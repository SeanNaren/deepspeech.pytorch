from argparse import Namespace
from enum import Enum
from typing import Union, Dict, Any

from pl_bolts.loggers import TrainsLogger
from pytorch_lightning.utilities import rank_zero_only


class DeepSpeechTrainsLogger(TrainsLogger):
    """
    Adds functionality to handle unique Enum classes in config dataclasses for Trains logging.
    """

    def _enums_convert(self, params):
        """
        Recursively checks parameters and converts Enums into strings for Trains to parse as hyperparams.
        :param params: Params dict
        :return: converted params dict
        """
        for k, v in params.items():
            if isinstance(v, dict):
                self._enums_convert(v)
            if isinstance(v, Enum):
                params[k] = str(v)  # Convert into string form for Trains parsing

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        """
        Log hyperparameters (numeric values) in TRAINS experiments.

        Args:
            params: The hyperparameters that passed through the model.
        """
        if not self._trains:
            return
        if not params:
            return

        params = self._convert_params(params)
        params = self._flatten_dict(params)
        self._enums_convert(params)
        self._trains.connect(params)
