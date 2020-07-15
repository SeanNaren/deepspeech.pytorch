import os
import shutil
import tempfile
import unittest
from dataclasses import dataclass

from data.an4 import download_an4
from deepspeech_pytorch.configs.inference_config import EvalConfig, ModelConfig, TranscribeConfig, LMConfig
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig, AdamConfig, BiDirectionalConfig, \
    FileCheckpointConfig, \
    DataConfig, TrainingConfig
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
                                   epoch: int,
                                   batch_size: int,
                                   model_config: BiDirectionalConfig,
                                   use_half: bool,
                                   cuda: bool):
        train_manifest, val_manifest, test_manifest = self.download_data(DatasetConfig(target_dir=self.target_dir,
                                                                                       manifest_dir=self.manifest_dir))

        train_cfg = self.create_training_config(epoch=epoch,
                                                batch_size=batch_size,
                                                train_manifest=train_manifest,
                                                val_manifest=val_manifest,
                                                model_config=model_config,
                                                cuda=cuda)
        print("Running Training DeepSpeech Model Smoke Test")
        train(train_cfg)

        # Expected final model path after training
        model_path = self.model_dir + '/deepspeech_final.pth'
        assert os.path.exists(model_path)

        lm_configs = [
            LMConfig(),  # Test Greedy
            LMConfig(
                decoder_type=DecoderType.beam
            )  # Test Beam Decoder
        ]
        print("Running Inference Smoke Tests")
        for lm_config in lm_configs:
            self.eval_model(
                model_path=model_path,
                test_manifest=test_manifest,
                cuda=cuda,
                use_half=use_half,
                lm_config=lm_config
            )

            self.inference(test_manifest=test_manifest,
                           model_path=model_path,
                           cuda=cuda,
                           use_half=use_half,
                           lm_config=lm_config)

    def eval_model(self,
                   model_path: str,
                   test_manifest: str,
                   cuda: bool,
                   use_half: bool,
                   lm_config: LMConfig):
        # Due to using TravisCI with no GPU support we have to disable cuda
        eval_cfg = EvalConfig(
            model=ModelConfig(
                cuda=cuda,
                model_path=model_path,
                use_half=use_half
            ),
            lm=lm_config,
            test_manifest=test_manifest
        )
        evaluate(eval_cfg)

    def inference(self,
                  test_manifest: str,
                  model_path: str,
                  cuda: bool,
                  use_half: bool,
                  lm_config: LMConfig):
        # Select one file from our test manifest to run inference
        with open(test_manifest) as f:
            file_path = next(f).strip().split(',')[0]

        transcribe_cfg = TranscribeConfig(
            model=ModelConfig(
                cuda=cuda,
                model_path=model_path,
                use_half=use_half
            ),
            lm=lm_config,
            audio_path=file_path
        )
        transcribe(transcribe_cfg)

    def download_data(self, cfg: DatasetConfig):
        download_an4(target_dir=cfg.target_dir,
                     manifest_dir=cfg.manifest_dir,
                     min_duration=cfg.min_duration,
                     max_duration=cfg.max_duration,
                     val_fraction=cfg.val_fraction,
                     sample_rate=cfg.sample_rate)
        # Expected manifests paths
        train_manifest = os.path.join(self.manifest_dir, 'an4_train_manifest.csv')
        val_manifest = os.path.join(self.manifest_dir, 'an4_val_manifest.csv')
        test_manifest = os.path.join(self.manifest_dir, 'an4_test_manifest.csv')

        # Assert manifest paths exists
        assert os.path.exists(train_manifest)
        assert os.path.exists(val_manifest)
        assert os.path.exists(test_manifest)
        return train_manifest, val_manifest, test_manifest

    def create_training_config(self,
                               epoch: int,
                               batch_size: int,
                               train_manifest: str,
                               val_manifest: str,
                               model_config: BiDirectionalConfig,
                               cuda: bool):
        return DeepSpeechConfig(
            training=TrainingConfig(epochs=epoch,
                                    no_cuda=not cuda),
            data=DataConfig(train_manifest=train_manifest,
                            val_manifest=val_manifest,
                            batch_size=batch_size),
            optim=AdamConfig(),
            model=model_config,
            checkpointing=FileCheckpointConfig(save_folder=self.model_dir)
        )


class AN4SmokeTest(DeepSpeechSmokeTest):

    def test_train_eval_inference(self):
        # Hardcoded sizes to reduce memory/time, and disabled GPU due to using TravisCI
        model_cfg = BiDirectionalConfig(hidden_size=10,
                                        hidden_layers=1)
        self.build_train_evaluate_model(epoch=1,
                                        batch_size=10,
                                        model_config=model_cfg,
                                        cuda=False,
                                        use_half=False)


if __name__ == '__main__':
    unittest.main()
