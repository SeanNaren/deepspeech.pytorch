import os
import unittest

import wget

from deepspeech_pytorch.configs.inference_config import LMConfig
from deepspeech_pytorch.enums import DecoderType
from tests.smoke_test import DatasetConfig, DeepSpeechSmokeTest

pretrained_urls = [
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/latest/download/an4_pretrained_v2.pth',
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/latest/download/librispeech_pretrained_v2.pth',
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/latest/download/ted_pretrained_v2.pth'
]

lm_path = 'http://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz'


class PretrainedSmokeTest(DeepSpeechSmokeTest):

    def test_pretrained_eval_inference(self):
        # Disabled GPU due to using TravisCI
        cuda, use_half = False, False
        train_manifest, val_manifest, test_manifest = self.download_data(DatasetConfig(target_dir=self.target_dir,
                                                                                       manifest_dir=self.manifest_dir))
        wget.download(lm_path)
        for pretrained_url in pretrained_urls:
            print("Running Pre-trained Smoke test for: ", pretrained_url)
            wget.download(pretrained_url)
            file_path = os.path.basename(pretrained_url)
            pretrained_path = os.path.abspath(file_path)

            lm_configs = [
                LMConfig(),  # Greedy
                LMConfig(
                    decoder_type=DecoderType.beam
                ),  # Test Beam Decoder
                LMConfig(
                    decoder_type=DecoderType.beam,
                    lm_path=os.path.basename(lm_path),
                    alpha=1,
                    beta=1
                )  # Test Beam Decoder with LM
            ]

            for lm_config in lm_configs:
                self.eval_model(model_path=pretrained_path,
                                test_manifest=test_manifest,
                                cuda=cuda,
                                use_half=use_half,
                                lm_config=lm_config)
                self.inference(test_manifest=test_manifest,
                               model_path=pretrained_path,
                               cuda=cuda,
                               lm_config=lm_config,
                               use_half=use_half)


if __name__ == '__main__':
    unittest.main()
