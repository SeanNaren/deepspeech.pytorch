import argparse
from pathlib import Path

from deepspeech_pytorch.utils import load_model
import torch

parser = argparse.ArgumentParser()
parser.add_argument("model", nargs="?", help="Path to model to convert")
parser.add_argument(
    "-d",
    "--device",
    type=str,
    choices=["cuda", "cpu"],
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="PyTorch device to use",
)
parser.add_argument("-m", "--method", type=str, choices=["script", "trace"], default="script",
                    help="Script or Trace model"
)
parser.add_argument("-o", "--output", type=str, default="./", help="Output directory")
args = parser.parse_args()


def main():
    print(f"Loading {args.model} on {args.device}")
    model = load_model(args.device, args.model)
    # First 3 sizes are constant (MFCC features). Last dimension is dynamic (Length/Size).
    example_spect = torch.rand((1, 1, 161, 493))  # Batch x Features
    example_input_size = torch.IntTensor([example_spect.size(3)]).int()
    example_input = (example_spect, example_input_size)
    print("Converting model...")
    model_script = model.to_torchscript(method=args.method, example_inputs=example_input)
    model_name = f"{Path(args.output).stem}_{args.method}_{args.device}.ckpt"
    output_path = Path(args.output) / model_name
    print(f"Saving to {output_path}")
    torch.jit.save(model_script, output_path.absolute().as_posix())


if __name__ == '__main__':
    main()
