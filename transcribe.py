import hydra

from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.inference import transcribe


@hydra.main(config_name="config")
def hydra_main(cfg: TranscribeConfig):
    transcribe(cfg=cfg)


if __name__ == '__main__':
    hydra_main()
