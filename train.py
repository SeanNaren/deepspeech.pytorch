from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI

from deepspeech_pytorch.data.datamodule import DeepSpeechDataModule
from deepspeech_pytorch.model import DeepSpeech


class DeepSpeechCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.config.spect.sample_rate", "model.sample_rate")
        parser.link_arguments("data.config.spect.window_size", "model.window_size")
        parser.link_arguments("data.labels", "model.labels")
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
        parser.set_defaults({"checkpoint.monitor": "wer", "checkpoint.verbose": True})


if __name__ == '__main__':
    DeepSpeechCLI(
        DeepSpeech,
        DeepSpeechDataModule,
        trainer_defaults=dict(replace_sampler_ddp=False)
    )
