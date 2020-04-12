import argparse
import json

parser = argparse.ArgumentParser(description='Select the best parameters based on the WER')
parser.add_argument('--input-path', type=str, help='Output json file from search_lm_params')
args = parser.parse_args()

with open(args.input_path) as f:
    results = json.load(f)

min_results = min(results, key=lambda x: x[2])  # Find the minimum WER (alpha, beta, WER, CER)
print("Alpha: %f \nBeta: %f \nWER: %f\nCER: %f" % tuple(min_results))
