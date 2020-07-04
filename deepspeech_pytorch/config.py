from dataclasses import dataclass, field
from typing import Any, List

from deepspeech_pytorch.enums import DistributedBackend, SpectrogramWindow, RNNType
from omegaconf import MISSING

defaults = [
    {"optim": "sgd"},
    {"model": "bidirectional"},
    {"checkpointing": "file"}
]


@dataclass
class TrainingConfig:
    no_cuda: bool = False  # Enable CPU only training
    finetune: bool = False  # Fine-tune the model from checkpoint "continue_from"
    seed: int = 123456  # Seed for generators
    dist_backend: DistributedBackend = DistributedBackend.nccl  # If using distribution, the backend to be used
    epochs: int = 70  # Number of Training Epochs


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
    train_manifest: str = 'data/train_manifest.csv'
    val_manifest: str = 'data/val_manifest.csv'
    batch_size: int = 20  # Batch size for training
    num_workers: int = 4  # Number of workers used in data-loading
    labels_path: str = 'labels.json'  # Contains tokens for model output
    spect: SpectConfig = SpectConfig()
    augmentation: AugmentationConfig = AugmentationConfig()


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
    learning_rate: float = 3e-4  # Initial Learning Rate
    learning_anneal: float = 1.1  # Annealing applied to learning rate after each epoch
    weight_decay: float = 1e-5  # Initial Weight Decay
    max_norm: float = 400  # Norm cutoff to prevent explosion of gradients


@dataclass
class SGDConfig(OptimConfig):
    momentum: float = 0.9


@dataclass
class AdamConfig(OptimConfig):
    eps: float = 1e-8  # Adam eps
    betas: tuple = (0.9, 0.999)  # Adam betas


@dataclass
class CheckpointConfig:
    continue_from: str = ''  # Continue training from checkpoint model
    checkpoint: bool = True  # Enables epoch checkpoint saving of model
    checkpoint_per_iteration: int = 0  # Save checkpoint per N number of iterations. Default is disabled
    save_n_recent_models: int = 10  # Max number of checkpoints to save, delete older checkpoints
    best_val_model_name: str = 'deepspeech_final.pth'  # Name to save best validated model within the save folder
    load_auto_checkpoint: bool = False  # Automatically load the latest checkpoint from save folder


@dataclass
class FileCheckpointConfig(CheckpointConfig):
    save_folder: str = 'models/'  # Location to save checkpoint models


@dataclass
class GCSCheckpointConfig(CheckpointConfig):
    gcs_bucket: str = MISSING  # Bucket to store model checkpoints e.g bucket-name
    gcs_save_folder: str = MISSING  # Folder to store model checkpoints in bucket e.g models/
    local_save_file: str = './local_checkpoint.pth'  # Place to store temp file on disk


@dataclass
class VisualizationConfig:
    id: str = 'DeepSpeech training'  # Name to use when visualizing/storing the run
    visdom: bool = False  # Turn on visdom graphing
    tensorboard: bool = False  # Turn on Tensorboard graphing
    log_dir: str = 'visualize/deepspeech_final'  # Location of Tensorboard log
    log_params: bool = False  # Log parameter values and gradients


@dataclass
class ApexConfig:
    opt_level: str = 'O1'  # Apex optimization level, check https://nvidia.github.io/apex/amp.html for more information
    loss_scale: int = 1  # Loss scaling used by Apex. Default is 1 due to warp-ctc not supporting scaling of gradients


@dataclass
class DeepSpeechConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    optim: Any = MISSING
    model: Any = MISSING
    checkpointing: Any = MISSING
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    apex: ApexConfig = ApexConfig()
    visualization: VisualizationConfig = VisualizationConfig()
