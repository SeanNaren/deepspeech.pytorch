import os
import unittest

import wget

from tests.smoke_test import DatasetConfig, DeepSpeechSmokeTest

pretrained_urls = [
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/latest/download/an4_pretrained_v2.pth',
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/latest/download/librispeech_pretrained_v2.pth',
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/latest/download/ted_pretrained_v2.pth'
]


class PretrainedSmokeTest(DeepSpeechSmokeTest):

    def test_pretrained_eval_inference(self):
        # Disabled GPU due to using TravisCI
        cuda, use_half = False, False
        train_manifest, val_manifest, test_manifest = self.download_data(DatasetConfig(target_dir=self.target_dir,
                                                                                       manifest_dir=self.manifest_dir))
        for pretrained_url in pretrained_urls:
            wget.download(pretrained_url)
            file_path = os.path.basename(pretrained_url)
            pretrained_path = os.path.abspath(file_path)
            self.eval_model(model_path=pretrained_path,
                            test_manifest=test_manifest,
                            cuda=cuda,
                            use_half=use_half)
            self.inference(test_manifest=test_manifest,
                           model_path=pretrained_path,
                           cuda=cuda,
                           use_half=use_half)


if __name__ == '__main__':
    unittest.main()
