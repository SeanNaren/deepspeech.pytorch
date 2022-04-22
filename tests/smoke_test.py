import json
import os
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from pytorch_lightning.utilities import _module_available

from data.an4 import download_an4
from deepspeech_pytorch.configs.inference_config import EvalConfig, ModelConfig, TranscribeConfig, LMConfig
from deepspeech_pytorch.configs.lightning_config import ModelCheckpointConf
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig, AdamConfig, BiDirectionalConfig, \
    DataConfig, DeepSpeechTrainerConf
from deepspeech_pytorch.enums import DecoderType
from deepspeech_pytorch.inference import transcribe
from deepspeech_pytorch.testing import evaluate
from deepspeech_pytorch.training import train


@dataclass
class DatasetConfig:
    target_dir: str = ''
    manifest_dir: str = ''
    min_duration: float = 0
    max_duration: float = 15
    val_fraction: float = 0.1
    sample_rate: int = 16000
    num_workers: int = 4


class DeepSpeechSmokeTest(unittest.TestCase):
    def setUp(self):
        self.target_dir = tempfile.mkdtemp()
        self.manifest_dir = tempfile.mkdtemp()
        self.model_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.target_dir)
        shutil.rmtree(self.manifest_dir)
        shutil.rmtree(self.model_dir)

    def build_train_evaluate_model(self,
                                   limit_train_batches: int,
                                   limit_val_batches: int,
                                   epoch: int,
                                   batch_size: int,
                                   model_config: BiDirectionalConfig,
                                   precision: int,
                                   gpus: int,
                                   folders: bool):
        cuda = gpus > 0

        train_path, val_path, test_path = self.download_data(
            DatasetConfig(
                target_dir=self.target_dir,
                manifest_dir=self.manifest_dir
            ),
            folders=folders
        )

        train_cfg = self.create_training_config(
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            max_epochs=epoch,
            batch_size=batch_size,
            train_path=train_path,
            val_path=val_path,
            model_config=model_config,
            precision=precision,
            gpus=gpus
        )
        print("Running Training DeepSpeech Model Smoke Test")
        train(train_cfg)

        # Expected final model path after training
        print(os.listdir(self.model_dir))
        model_path = self.model_dir + '/last.ckpt'
        assert os.path.exists(model_path)

        lm_configs = [LMConfig()]

        if _module_available('ctcdecode'):
            lm_configs.append(
                LMConfig(
                    decoder_type=DecoderType.beam
                )
            )
        print("Running Inference Smoke Tests")
        for lm_config in lm_configs:
            self.eval_model(
                model_path=model_path,
                test_path=test_path,
                cuda=cuda,
                precision=precision,
                lm_config=lm_config
            )

            self.inference(test_path=test_path,
                           model_path=model_path,
                           cuda=cuda,
                           precision=precision,
                           lm_config=lm_config)

    def eval_model(self,
                   model_path: str,
                   test_path: str,
                   cuda: bool,
                   precision: int,
                   lm_config: LMConfig):
        # Due to using TravisCI with no GPU support we have to disable cuda
        eval_cfg = EvalConfig(
            model=ModelConfig(
                cuda=cuda,
                model_path=model_path,
                precision=precision
            ),
            lm=lm_config,
            test_path=test_path
        )
        evaluate(eval_cfg)

    def inference(self,
                  test_path: str,
                  model_path: str,
                  cuda: bool,
                  precision: int,
                  lm_config: LMConfig):
        # Select one file from our test manifest to run inference
        if os.path.isdir(test_path):
            file_path = next(Path(test_path).rglob('*.wav'))
        else:
            with open(test_path) as f:
                # select a file to use for inference test
                manifest = json.load(f)
                file_name = manifest['samples'][0]['wav_path']
                directory = manifest['root_path']
                file_path = os.path.join(directory, file_name)

        transcribe_cfg = TranscribeConfig(
            model=ModelConfig(
                cuda=cuda,
                model_path=model_path,
                precision=precision
            ),
            lm=lm_config,
            audio_path=file_path
        )
        transcribe(transcribe_cfg)

    def download_data(self,
                      cfg: DatasetConfig,
                      folders: bool):
        download_an4(
            target_dir=cfg.target_dir,
            manifest_dir=cfg.manifest_dir,
            min_duration=cfg.min_duration,
            max_duration=cfg.max_duration,
            num_workers=cfg.num_workers
        )

        # Expected output paths
        if folders:
            train_path = os.path.join(self.target_dir, 'train/')
            val_path = os.path.join(self.target_dir, 'val/')
            test_path = os.path.join(self.target_dir, 'test/')
        else:
            train_path = os.path.join(self.manifest_dir, 'an4_train_manifest.json')
            val_path = os.path.join(self.manifest_dir, 'an4_val_manifest.json')
            test_path = os.path.join(self.manifest_dir, 'an4_test_manifest.json')

        # Assert manifest paths exists
        assert os.path.exists(train_path)
        assert os.path.exists(val_path)
        assert os.path.exists(test_path)
        return train_path, val_path, test_path

    def create_training_config(self,
                               limit_train_batches: int,
                               limit_val_batches: int,
                               max_epochs: int,
                               batch_size: int,
                               train_path: str,
                               val_path: str,
                               model_config: BiDirectionalConfig,
                               precision: int,
                               gpus: int):
        return DeepSpeechConfig(
            trainer=DeepSpeechTrainerConf(
                max_epochs=max_epochs,
                precision=precision,
                gpus=gpus,
                enable_checkpointing=True,
                limit_train_batches=limit_train_batches,
                limit_val_batches=limit_val_batches
            ),
            data=DataConfig(
                train_path=train_path,
                val_path=val_path,
                batch_size=batch_size
            ),
            optim=AdamConfig(),
            model=model_config,
            checkpoint=ModelCheckpointConf(
                dirpath=self.model_dir,
                save_last=True,
                verbose=True
            )
        )


class AN4SmokeTest(DeepSpeechSmokeTest):

    def test_train_eval_inference(self):
        # Hardcoded sizes to reduce memory/time, and disabled GPU due to using TravisCI
        model_cfg = BiDirectionalConfig(
            hidden_size=10,
            hidden_layers=1
        )
        self.build_train_evaluate_model(
            limit_train_batches=1,
            limit_val_batches=1,
            epoch=1,
            batch_size=10,
            model_config=model_cfg,
            precision=32,
            gpus=0,
            folders=False
        )

    def test_train_eval_inference_folder(self):
        """Test train/eval/inference using folder directories rather than manifest files"""
        # Hardcoded sizes to reduce memory/time, and disabled GPU due to using TravisCI
        model_cfg = BiDirectionalConfig(
            hidden_size=10,
            hidden_layers=1
        )
        self.build_train_evaluate_model(
            limit_train_batches=1,
            limit_val_batches=1,
            epoch=1,
            batch_size=10,
            model_config=model_cfg,
            precision=32,
            gpus=0,
            folders=True
        )


if __name__ == '__main__':
    unittest.main()
