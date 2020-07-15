import hydra
import torch
from tqdm import tqdm

from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.utils import load_model, load_decoder


@torch.no_grad()
def evaluate(cfg: EvalConfig):
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model = load_model(device=device,
                       model_path=cfg.model.model_path,
                       use_half=cfg.model.use_half)

    decoder = load_decoder(labels=model.labels,
                           cfg=cfg.lm)
    target_decoder = GreedyDecoder(model.labels,
                                   blank_index=model.labels.index('_'))
    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                      manifest_filepath=hydra.utils.to_absolute_path(cfg.test_manifest),
                                      labels=model.labels,
                                      normalize=True)
    test_loader = AudioDataLoader(test_dataset,
                                  batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers)
    wer, cer, output_data = run_evaluation(test_loader=test_loader,
                                           device=device,
                                           model=model,
                                           decoder=decoder,
                                           target_decoder=target_decoder,
                                           save_output=cfg.save_output,
                                           verbose=cfg.verbose,
                                           use_half=cfg.model.use_half)

    print('Test Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
    if cfg.save_output:
        torch.save(output_data, hydra.utils.to_absolute_path(cfg.save_output))


@torch.no_grad()
def run_evaluation(test_loader,
                   device,
                   model,
                   decoder,
                   target_decoder,
                   save_output=None,
                   verbose=False,
                   use_half=False):
    model.eval()
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        if use_half:
            inputs = inputs.half()
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        out, output_sizes = model(inputs, input_sizes)

        decoded_output, _ = decoder.decode(out, output_sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)

        if save_output is not None:
            # add output to data array, and continue
            output_data.append((out.cpu(), output_sizes, target_strings))
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = decoder.wer(transcript, reference)
            cer_inst = decoder.cer(transcript, reference)
            total_wer += wer_inst
            total_cer += cer_inst
            num_tokens += len(reference.split())
            num_chars += len(reference.replace(' ', ''))
            if verbose:
                print("Ref:", reference.lower())
                print("Hyp:", transcript.lower())
                print("WER:", float(wer_inst) / len(reference.split()),
                      "CER:", float(cer_inst) / len(reference.replace(' ', '')), "\n")
    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars
    return wer * 100, cer * 100, output_data
