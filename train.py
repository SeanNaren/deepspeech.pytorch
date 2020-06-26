import hydra
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from deepspeech_pytorch.loader.data_loader import SpectrogramWindow
from deepspeech_pytorch.training import train, DistributedBackend


@dataclass
class TrainConfig:
    no_cuda: bool = False  # Enable CPU only training
    finetune: bool = False  # Fine-tune the model from checkpoint "continue_from"
    seed: int = 123456  # Seed for generators
    dist_backend: DistributedBackend = DistributedBackend.nccl  # If using distribution, the backend to be used
    epochs: int = 70  # Number of Training Epochs


@dataclass
class DataConfig:
    train_manifest: str = 'data/train_manifest.csv'
    val_manifest: str = 'data/val_manifest.csv'
    sample_rate: int = 16000  # The sample rate for the data/model features
    batch_size: int = 20  # Batch size for training
    num_workers: int = 4  # Number of workers used in data-loading
    labels_path: str = 'labels.json'  # Contains tokens for model output
    window_size: float = .02  # Window size for spectrogram generation (seconds)
    window_stride: float = .01  # Window stride for spectrogram generation (seconds)
    window: SpectrogramWindow = SpectrogramWindow.hamming  # Window type for spectrogram generation


@dataclass
class ModelConfig:
    rnn_type: str = 'lstm'  # Type of RNN to use in modeel, rnn/gru/lstm are supported
    hidden_size: int = 1024  # Hidden size of RNN Layer
    hidden_layers: int = 5  # Number of RNN layers
    bidirectional: bool = True  # Use BiRNNs. If False, uses lookahead conv


@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4  # Initial Learning Rate
    weight_decay: float = 1e-5  # Initial Weight Decay
    momentum: float = 0.9
    adam: bool = False  # Replace SGD with AdamW
    eps: float = 1e-8  # Adam eps
    beta: tuple = (0.9, 0.999)  # Adam betas
    max_norm: float = 400  # Norm cutoff to prevent explosion of gradients
    learning_anneal: float = 1.1  # Annealing applied to learning rate after each epoch


@dataclass
class CheckpointConfig:
    continue_from: str = ''  # Continue training from checkpoint model
    checkpoint: bool = True  # Enables epoch checkpoint saving of model
    checkpoint_per_iteration: int = 0  # Save checkpoint per N number of iterations. Default is disabled
    save_n_recent_models: int = 10  # Max number of checkpoints to save, delete older checkpoints
    save_folder: str = 'models/'  # Location to save epoch models
    best_val_model_name: str = 'deepspeech_final.pth'  # Name to save best validated model within the save folder
    load_auto_checkpoint: bool = False  # Automatically load the latest checkpoint from save folder


@dataclass
class VisualizationConfig:
    visdom: bool = False  # Turn on visdom graphing
    tensorboard: bool = False  # Turn on Tensorboard graphing
    log_dir: str = 'visualize/deepspeech_final'  # Location of Tensorboard log
    log_params: bool = False  # Log parameter values and gradients
    id: str = 'DeepSpeech training'  # Identifier for visdom/tensorboard run


@dataclass
class AugmentationConfig:
    speed_volume_perturb: bool = False  # Use random tempo and gain perturbations.
    spec_augment: bool = False  # Use simple spectral augmentation on mel spectograms.
    noise_dir: str = ''  # Directory to inject noise into audio. If default, noise Inject not added
    noise_prob: float = 0.4  # Probability of noise being added per sample
    noise_min: float = 0.0  # Minimum noise level to sample from. (1.0 means all noise, not original signal)
    noise_max: float = 0.5  # Maximum noise levels to sample from. Maximum 1.0


@dataclass
class ApexConfig:
    opt_level: str = 'O1'  # Apex optimization level, check https://nvidia.github.io/apex/amp.html for more information
    loss_scale: int = 1  # Loss scaling used by Apex. Default is 1 due to warp-ctc not supporting scaling of gradients


@dataclass
class DeepSpeechConfig(DictConfig):
    training: TrainConfig = TrainConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    checkpointing: CheckpointConfig = CheckpointConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    apex: ApexConfig = ApexConfig()


cs = ConfigStore.instance()
cs.store(name="config", node=DeepSpeechConfig)


@hydra.main(config_name="config")
def hydra_main(cfg: DeepSpeechConfig):
    train(cfg=cfg)


if __name__ == '__main__':
    hydra_main()
