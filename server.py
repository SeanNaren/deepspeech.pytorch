import os
from tempfile import NamedTemporaryFile

import torch
from flask import Flask, request, jsonify
import logging
from data.data_loader import SpectrogramParser
from decoder import GreedyDecoder
from model import DeepSpeech
from opts import add_decoder_args, add_inference_args
from transcribe import transcribe

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['.wav', '.mp3', '.ogg', '.webm'])


@app.route('/transcribe', methods=['POST'])
def transcribe_file():
    if request.method == 'POST':
        res = {}
        if 'file' not in request.files:
            res['status'] = "error"
            res['message'] = "audio file should be passed for the transcription"
            return jsonify(res)
        file = request.files['file']
        filename = file.filename
        _, file_extension = os.path.splitext(filename)
        if file_extension.lower() not in ALLOWED_EXTENSIONS:
            res['status'] = "error"
            res['message'] = "{} is not supported format.".format(file_extension)
            return jsonify(res)
        with NamedTemporaryFile(suffix=file_extension) as tmp_saved_audio_file:
            file.save(tmp_saved_audio_file.name)
            logging.info('Transcribing file...')
            transcription, _ = transcribe(tmp_saved_audio_file.name, spect_parser, model, decoder, args.cuda)
            logging.info('File transcribed')
            res['status'] = "OK"
            res['transcription'] = transcription
            return jsonify(res)


def main():
    import argparse
    global model, spect_parser, decoder, args
    parser = argparse.ArgumentParser(description='DeepSpeech transcription server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to be used by the server')
    parser.add_argument('--port', type=int, default=8888, help='Port to be used by the server')
    parser = add_inference_args(parser)
    parser = add_decoder_args(parser)
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info('Setting up server...')
    torch.set_grad_enabled(False)
    model = DeepSpeech.load_model(args.model_path)
    if args.cuda:
        model.cuda()
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    else:
        decoder = GreedyDecoder(labels, blank_index=labels.index('_'))

    spect_parser = SpectrogramParser(audio_conf, normalize=True)
    logging.info('Server initialised')
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
