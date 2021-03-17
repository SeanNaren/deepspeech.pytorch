from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
import optuna
import torch

from deepspeech_pytorch.configs.train_config import SpectConfig
from deepspeech_pytorch.decoder import BeamCTCDecoder, GreedyDecoder
from deepspeech_pytorch.loader.data_loader import AudioDataLoader, SpectrogramDataset
from deepspeech_pytorch.utils import load_model
from deepspeech_pytorch.validation import run_evaluation


@dataclass
class OptimizerConfig:
    model_path: str = ''
    test_path: str = ''  # Path to test manifest or csv
    is_character_based: bool = True  # Use CER or WER for finding optimal parameters
    lm_path: str = ''
    beam_width: int = 10
    alpha_from: float = 0.0
    alpha_to: float = 3.0
    beta_from: float = 0.0
    beta_to: float = 1.0
    n_trials: int = 500  # Number of trials for optuna
    n_jobs: int = 2      # Number of parallel jobs for optuna
    precision: int = 16
    batch_size: int = 1   # For dataloader
    num_workers: int = 1  # For dataloader
    spect_cfg: SpectConfig = SpectConfig()


cs = ConfigStore.instance()
cs.store(name="config", node=OptimizerConfig)


class Objective(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_model(
            self.device,
            hydra.utils.to_absolute_path(self.cfg.model_path)
        )
        self.ckpt = torch.load(
            hydra.utils.to_absolute_path(self.cfg.model_path),
            map_location=self.device
        )
        self.labels = self.ckpt['hyper_parameters']['labels']

        self.decoder = BeamCTCDecoder(
            labels=self.labels,
            lm_path=hydra.utils.to_absolute_path(self.cfg.lm_path),
            beam_width=self.cfg.beam_width,
            num_processes=self.cfg.num_workers,
            blank_index=self.labels.index('_')
        )
        self.target_decoder = GreedyDecoder(
            labels=self.labels,
            blank_index=self.labels.index('_')
        )

        test_dataset = SpectrogramDataset(
            audio_conf=self.cfg.spect_cfg,
            input_path=hydra.utils.to_absolute_path(cfg.test_path),
            labels=self.labels,
            normalize=True
        )
        self.test_loader = AudioDataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers
        )

    def __call__(self, trial):
        alpha = trial.suggest_uniform('alpha', self.cfg.alpha_from, self.cfg.alpha_to)
        beta = trial.suggest_uniform('beta', self.cfg.beta_from, self.cfg.beta_to)
        self.decoder._decoder.reset_params(alpha, beta)

        wer, cer = run_evaluation(
            test_loader=self.test_loader,
            device=self.device,
            model=self.model,
            decoder=self.decoder,
            target_decoder=self.target_decoder,
            precision=self.cfg.precision
        )
        return cer if self.cfg.is_character_based else wer


@hydra.main(config_name="config")
def main(cfg: OptimizerConfig) -> None:
    study = optuna.create_study()
    study.optimize(Objective(cfg),
                   n_trials=cfg.n_trials,
                   n_jobs=cfg.n_jobs,
                   show_progress_bar=True)
    print(f"Best Params\n"
          f"alpha: {study.best_params['alpha']}\n"
          f"beta: {study.best_params['beta']}\n"
          f"{'cer' if cfg.is_character_based else 'wer'}: {study.best_value}")


if __name__ == "__main__":
    main()
