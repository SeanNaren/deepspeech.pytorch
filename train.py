import hydra
from hydra.core.config_store import ConfigStore

from deepspeech_pytorch.config import DeepSpeechConfig, AdamConfig, SGDConfig
from deepspeech_pytorch.training import train

cs = ConfigStore.instance()
cs.store(group="optim", name="sgd", node=SGDConfig)
cs.store(group="optim", name="adam", node=AdamConfig)
cs.store(name="config", node=DeepSpeechConfig)


@hydra.main(config_name="config")
def hydra_main(cfg: DeepSpeechConfig):
    train(cfg=cfg)


if __name__ == '__main__':
    hydra_main()
