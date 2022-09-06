from dataclasses import dataclass, field
from typing import Any, List

from omegaconf import MISSING

from deepspeech_pytorch.configs.lightning_config import TrainerConf, ModelCheckpointConf
from deepspeech_pytorch.enums import SpectrogramWindow, RNNType

defaults = [
    {"optim": "adam"},
    {"model": "bidirectional"},
    {"checkpoint": "file"}
]


@dataclass
class SpectConfig:
    sample_rate: int = 16000  # The sample rate for the data/model features
    window_size: float = .02  # Window size for spectrogram generation (seconds)
    window_stride: float = .01  # Window stride for spectrogram generation (seconds)
    window: SpectrogramWindow = SpectrogramWindow.hamming  # Window type for spectrogram generation


@dataclass
class AugmentationConfig:
    speed_volume_perturb: bool = False  # Use random tempo and gain perturbations.
    spec_augment: bool = False  # Use simple spectral augmentation on mel spectograms.
    noise_dir: str = ''  # Directory to inject noise into audio. If default, noise Inject not added
    noise_prob: float = 0.4  # Probability of noise being added per sample
    noise_min: float = 0.0  # Minimum noise level to sample from. (1.0 means all noise, not original signal)
    noise_max: float = 0.5  # Maximum noise levels to sample from. Maximum 1.0


@dataclass
class DataConfig:
    train_path: str = 'data/train_manifest.csv'
    val_path: str = 'data/val_manifest.csv'
    batch_size: int = 64  # Batch size for training
    num_workers: int = 4  # Number of workers used in data-loading
    labels_path: str = 'labels.json'  # Contains tokens for model output
    spect: SpectConfig = SpectConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    prepare_data_per_node: bool = True


@dataclass
class BiDirectionalConfig:
    rnn_type: RNNType = RNNType.lstm  # Type of RNN to use in model
    hidden_size: int = 1024  # Hidden size of RNN Layer
    hidden_layers: int = 5  # Number of RNN layers


@dataclass
class UniDirectionalConfig(BiDirectionalConfig):
    lookahead_context: int = 20  # The lookahead context for convolution after RNN layers


@dataclass
class OptimConfig:
    learning_rate: float = 1.5e-4  # Initial Learning Rate
    learning_anneal: float = 0.99  # Annealing applied to learning rate after each epoch
    weight_decay: float = 1e-5  # Initial Weight Decay


@dataclass
class SGDConfig(OptimConfig):
    momentum: float = 0.9


@dataclass
class AdamConfig(OptimConfig):
    eps: float = 1e-8  # Adam eps
    betas: tuple = (0.9, 0.999)  # Adam betas


@dataclass
class DeepSpeechTrainerConf(TrainerConf):
    callbacks: Any = MISSING


@dataclass
class DeepSpeechConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    optim: Any = MISSING
    model: Any = MISSING
    checkpoint: ModelCheckpointConf = MISSING
    trainer: DeepSpeechTrainerConf = DeepSpeechTrainerConf()
    data: DataConfig = DataConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    seed: int = 123456  # Seed for generators
    load_auto_checkpoint: bool = False  # Automatically load the latest checkpoint from save folder
