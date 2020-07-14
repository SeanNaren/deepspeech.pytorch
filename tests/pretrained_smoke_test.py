import os
import unittest

import wget

from tests.smoke_test import DatasetConfig, DeepSpeechSmokeTest

pretrained_urls = [
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/an4_pretrained_v2.pth',
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/librispeech_pretrained_v2.pth',
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/ted_pretrained_v2.pth'
]


class PretrainedSmokeTest(DeepSpeechSmokeTest):

    def test_pretrained_eval_inference(self):
        train_manifest, val_manifest, test_manifest = self.download_data(DatasetConfig(target_dir=self.target_dir,
                                                                                       manifest_dir=self.manifest_dir))
        for pretrained_url in pretrained_urls:
            wget.download(pretrained_url)
            file_path = os.path.basename(pretrained_url)
            pretrained_path = os.path.abspath(file_path)
            self.eval_model(model_path=pretrained_path,
                            test_manifest=test_manifest)

            with open(test_manifest) as f:
                test_file_paths = [x.strip().split(',')[0] for x in f]

            self.inference(test_file_paths=test_file_paths,
                           model_path=pretrained_path)


if __name__ == '__main__':
    unittest.main()
