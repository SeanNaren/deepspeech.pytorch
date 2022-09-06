from dataclasses import dataclass, field
from typing import Any
from typing import Optional


@dataclass
class ModelCheckpointConf:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    filepath: Optional[str] = None
    monitor: Optional[str] = None
    verbose: bool = False
    save_last: Optional[bool] = None
    save_top_k: Optional[int] = 1
    save_weights_only: bool = False
    mode: str = "min"
    dirpath: Any = None  # Union[str, Path, NoneType]
    filename: Optional[str] = None
    auto_insert_metric_name: bool = True
    every_n_train_steps: Optional[int] = None
    train_time_interval: Optional[str] = None
    every_n_epochs: Optional[int] = None
    save_on_train_epoch_end: Optional[bool] = None


@dataclass
class TrainerConf:
    _target_: str = "pytorch_lightning.trainer.Trainer"
    logger: Any = (
        True  # Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]
    )
    enable_checkpointing: bool = True
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0
    callbacks: Any = None
    num_nodes: int = 1
    num_processes: int = 1
    gpus: Any = None  # Union[int, str, List[int], NoneType]
    auto_select_gpus: bool = False
    tpu_cores: Any = None  # Union[int, str, List[int], NoneType]
    overfit_batches: Any = 0.0  # Union[int, float]
    track_grad_norm: Any = -1  # Union[int, float, str]
    check_val_every_n_epoch: int = 1
    fast_dev_run: Any = False  # Union[int, bool]
    accumulate_grad_batches: Any = 1  # Union[int, Dict[int, int], List[list]]
    max_epochs: int = 1000
    min_epochs: int = 1
    limit_train_batches: Any = 1.0  # Union[int, float]
    limit_val_batches: Any = 1.0  # Union[int, float]
    limit_test_batches: Any = 1.0  # Union[int, float]
    val_check_interval: Any = 1.0  # Union[int, float]
    log_every_n_steps: int = 50
    accelerator: Any = None  # Union[str, Accelerator, NoneType]
    sync_batchnorm: bool = False
    precision: int = 32
    weights_save_path: Optional[str] = None
    num_sanity_val_steps: int = 2
    resume_from_checkpoint: Any = None  # Union[str, Path, NoneType]
    profiler: Any = None  # Union[BaseProfiler, bool, str, NoneType]
    benchmark: bool = False
    deterministic: bool = False
    auto_lr_find: Any = False  # Union[bool, str]
    replace_sampler_ddp: bool = True
    detect_anomaly: bool = False
    auto_scale_batch_size: Any = False  # Union[str, bool]
    plugins: Any = None  # Union[str, list, NoneType]
    amp_backend: str = "native"
    amp_level: Any = None
    move_metrics_to_cpu: bool = False
    gradient_clip_algorithm: Optional[str] = None
    devices: Any = None
    ipus: Optional[int] = None
    enable_progress_bar: bool = True
    max_time: Optional[str] = None
    limit_predict_batches: float = 1.0
    strategy: Optional[str] = None
    enable_model_summary: bool = True
    reload_dataloaders_every_n_epochs: int = 0
    multiple_trainloader_mode: str = "max_size_cycle"
