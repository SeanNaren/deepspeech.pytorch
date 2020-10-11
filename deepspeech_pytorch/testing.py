import hydra
import torch

from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.utils import load_model, load_decoder
from deepspeech_pytorch.validation import run_evaluation


@torch.no_grad()
def evaluate(cfg: EvalConfig):
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model = load_model(
        device=device,
        model_path=cfg.model.model_path
    )

    decoder = load_decoder(
        labels=model.labels,
        cfg=cfg.lm
    )
    target_decoder = GreedyDecoder(
        labels=model.labels,
        blank_index=model.labels.index('_')
    )
    test_dataset = SpectrogramDataset(
        audio_conf=model.spect_cfg,
        input_path=hydra.utils.to_absolute_path(cfg.test_path),
        labels=model.labels,
        normalize=True
    )
    test_loader = AudioDataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )
    wer, cer = run_evaluation(
        test_loader=test_loader,
        device=device,
        model=model,
        decoder=decoder,
        target_decoder=target_decoder,
        precision=cfg.model.precision
    )

    print('Test Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
