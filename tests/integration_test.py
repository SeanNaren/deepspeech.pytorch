import os
import shutil
import tempfile
import unittest
from dataclasses import dataclass

from data.an4 import download_an4
from deepspeech_pytorch.config import DeepSpeechConfig, AdamConfig, BiDirectionalConfig, FileCheckpointConfig, \
    DataConfig, TrainingConfig
from deepspeech_pytorch.training import train


@dataclass
class DatasetConfig:
    target_dir: str = ''
    manifest_dir: str = ''
    min_duration: float = 0
    max_duration: float = 15
    val_fraction: float = 0.1
    sample_rate: int = 16000


class AN4IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.target_dir = tempfile.mkdtemp()
        self.manifest_dir = tempfile.mkdtemp()
        self.model_dir = tempfile.mkdtemp()
        self.download_data(DatasetConfig(target_dir=self.target_dir,
                                         manifest_dir=self.manifest_dir))
        self.train_manifest = os.path.join(self.manifest_dir, 'an4_train_manifest.csv')
        self.val_manifest = os.path.join(self.manifest_dir, 'an4_val_manifest.csv')
        self.test_manifest = os.path.join(self.manifest_dir, 'an4_test_manifest.csv')
        print(os.path.exists(self.train_manifest))

    def test_train(self):
        train(self.config(epoch=1, batch_size=10))

    def test_train_eval(self):
        train(self.config(epoch=1, batch_size=10))
        self.eval_model()

    def test_train_eval_inference(self):
        train(self.config(epoch=1, batch_size=10))
        self.eval_model()
        self.inference()

    def eval_model(self):
        # Define evaluation of the model using greedy decoding. using evaluate function
        pass

    def tearDown(self):
        shutil.rmtree(self.target_dir)
        shutil.rmtree(self.manifest_dir)
        shutil.rmtree(self.model_dir)

    def download_data(self, cfg: DatasetConfig):
        download_an4(target_dir=cfg.target_dir,
                     manifest_dir=cfg.manifest_dir,
                     min_duration=cfg.min_duration,
                     max_duration=cfg.max_duration,
                     val_fraction=cfg.val_fraction,
                     sample_rate=cfg.sample_rate)

    def config(self, epoch, batch_size):
        return DeepSpeechConfig(
            training=TrainingConfig(epochs=epoch,
                                    no_cuda=True),
            data=DataConfig(train_manifest=self.train_manifest,
                            val_manifest=self.val_manifest,
                            batch_size=batch_size),
            optim=AdamConfig(),
            model=BiDirectionalConfig(hidden_size=10,
                                      hidden_layers=2),
            checkpointing=FileCheckpointConfig(save_folder=self.model_dir)
        )


if __name__ == '__main__':
    unittest.main()
