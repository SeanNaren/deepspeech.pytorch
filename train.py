import hydra
from omegaconf import DictConfig

from deepspeech_pytorch.training import train


@hydra.main(config_path="config", strict=False)
def hydra_main(cfg: DictConfig):
    train(cfg=cfg)


if __name__ == '__main__':
    hydra_main()
