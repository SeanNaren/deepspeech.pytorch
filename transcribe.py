import hydra
from hydra.core.config_store import ConfigStore

from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.inference import transcribe

cs = ConfigStore.instance()
cs.store(name="config", node=TranscribeConfig)


@hydra.main(config_path='.', config_name="config")
def hydra_main(cfg: TranscribeConfig):
    transcribe(cfg=cfg)


if __name__ == '__main__':
    hydra_main()
