import hydra

from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.testing import evaluate


@hydra.main(config_name="config")
def hydra_main(cfg: EvalConfig):
    evaluate(cfg=cfg)


if __name__ == '__main__':
    hydra_main()
