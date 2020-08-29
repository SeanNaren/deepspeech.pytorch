import torch
from torch.cuda.amp import autocast
from tqdm import tqdm


@torch.no_grad()
def run_evaluation(test_loader,
                   model,
                   decoder,
                   device,
                   target_decoder,
                   save_output=False,
                   verbose=False,
                   use_half=False):
    model.eval()
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    for i, (batch) in tqdm(enumerate(test_loader), total=len(test_loader)):
        wer, cer, n_tokens, n_chars, model_output = run_validation_step(
            batch=batch,
            decoder=decoder,
            device=device,
            model=model,
            target_decoder=target_decoder,
            verbose=verbose,
            use_half=use_half
        )
        total_wer += wer
        total_cer += cer
        num_tokens += n_tokens
        num_chars += n_chars
        if save_output:
            output_data.append(model_output)
    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars
    return wer * 100, cer * 100, output_data


def run_validation_step(batch,
                        device,
                        decoder,
                        model,
                        target_decoder,
                        verbose,
                        use_half):
    inputs, targets, input_percentages, target_sizes = batch
    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
    inputs = inputs.to(device)

    # unflatten targets
    split_targets = []
    offset = 0
    for size in target_sizes:
        split_targets.append(targets[offset:offset + size])
        offset += size
    with autocast(enabled=use_half):
        out, output_sizes = model(inputs, input_sizes)
    decoded_output, _ = decoder.decode(out, output_sizes)
    target_strings = target_decoder.convert_to_strings(split_targets)
    wer, cer, n_tokens, n_chars = 0, 0, 0, 0
    for x in range(len(target_strings)):
        transcript, reference = decoded_output[x][0], target_strings[x][0]
        wer_inst = decoder.wer(transcript, reference)
        cer_inst = decoder.cer(transcript, reference)
        wer += wer_inst
        cer += cer_inst
        n_tokens += len(reference.split())
        n_chars += len(reference.replace(' ', ''))
        if verbose:
            print("Ref:", reference.lower())
            print("Hyp:", transcript.lower())
            print("WER:", float(wer_inst) / len(reference.split()),
                  "CER:", float(cer_inst) / len(reference.replace(' ', '')), "\n")
    return wer, cer, n_tokens, n_chars, (out.cpu(), output_sizes, target_strings)
