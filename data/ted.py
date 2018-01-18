import os
import wget
import tarfile
import argparse
import subprocess
import unicodedata
import io
from utils import create_manifest
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Processes and downloads TED-LIUMv2 dataset.')
parser.add_argument("--target-dir", default='TEDLIUM_dataset/', type=str, help="Directory to store the dataset.")
parser.add_argument("--tar-path", type=str, help="Path to the TEDLIUM_release tar if downloaded (Optional).")
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--min-duration', default=1, type=int,
                    help='Prunes training samples shorter than the min duration (given in seconds, default 1)')
parser.add_argument('--max-duration', default=15, type=int,
                    help='Prunes training samples longer than the max duration (given in seconds, default 15)')
args = parser.parse_args()

TED_LIUM_V2_DL_URL = "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz"


def get_utterances_from_stm(stm_file):
    """
    Return list of entries containing phrase and its start/end timings
    :param stm_file:
    :return:
    """
    res = []
    with io.open(stm_file, "r", encoding='utf-8') as f:
        for stm_line in f:
            tokens = stm_line.split()
            start_time = float(tokens[3])
            end_time = float(tokens[4])
            filename = tokens[0]
            transcript = unicodedata.normalize("NFKD",
                                               " ".join(t for t in tokens[6:]).strip()). \
                encode("utf-8", "ignore").decode("utf-8", "ignore")
            if transcript != "ignore_time_segment_in_scoring":
                res.append({
                    "start_time": start_time, "end_time": end_time,
                    "filename": filename, "transcript": transcript
                })
        return res


def cut_utterance(src_sph_file, target_wav_file, start_time, end_time, sample_rate=16000):
    subprocess.call(["sox {}  -r {} -b 16 -c 1 {} trim {} ={}".format(src_sph_file, str(sample_rate),
                                                                      target_wav_file, start_time, end_time)],
                    shell=True)


def _preprocess_transcript(phrase):
    return phrase.strip().upper()


def filter_short_utterances(utterance_info, min_len_sec=1.0):
    return utterance_info["end_time"] - utterance_info["start_time"] > min_len_sec


def prepare_dir(ted_dir):
    converted_dir = os.path.join(ted_dir, "converted")
    # directories to store converted wav files and their transcriptions
    wav_dir = os.path.join(converted_dir, "wav")
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
    txt_dir = os.path.join(converted_dir, "txt")
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    counter = 0
    entries = os.listdir(os.path.join(ted_dir, "sph"))
    for sph_file in tqdm(entries, total=len(entries)):
        speaker_name = sph_file.split('.sph')[0]

        sph_file_full = os.path.join(ted_dir, "sph", sph_file)
        stm_file_full = os.path.join(ted_dir, "stm", "{}.stm".format(speaker_name))

        assert os.path.exists(sph_file_full) and os.path.exists(stm_file_full)
        all_utterances = get_utterances_from_stm(stm_file_full)

        all_utterances = filter(filter_short_utterances, all_utterances)
        for utterance_id, utterance in enumerate(all_utterances):
            target_wav_file = os.path.join(wav_dir, "{}_{}.wav".format(utterance["filename"], str(utterance_id)))
            target_txt_file = os.path.join(txt_dir, "{}_{}.txt".format(utterance["filename"], str(utterance_id)))
            cut_utterance(sph_file_full, target_wav_file, utterance["start_time"], utterance["end_time"],
                          sample_rate=args.sample_rate)
            with io.FileIO(target_txt_file, "w") as f:
                f.write(_preprocess_transcript(utterance["transcript"]).encode('utf-8'))
        counter += 1


def main():
    target_dl_dir = args.target_dir
    if not os.path.exists(target_dl_dir):
        os.makedirs(target_dl_dir)

    target_unpacked_dir = os.path.join(target_dl_dir, "TEDLIUM_release2")
    if args.tar_path and os.path.exists(args.tar_path):
        target_file = args.tar_path
    else:
        print("Could not find downloaded TEDLIUM archive, Downloading corpus...")
        wget.download(TED_LIUM_V2_DL_URL, target_dl_dir)
        target_file = os.path.join(target_dl_dir, "TEDLIUM_release2.tar.gz")

    if not os.path.exists(target_unpacked_dir):
        print("Unpacking corpus...")
        tar = tarfile.open(target_file)
        tar.extractall(target_dl_dir)
        tar.close()
    else:
        print("Found TEDLIUM directory, skipping unpacking of tar files")

    train_ted_dir = os.path.join(target_unpacked_dir, "train")
    val_ted_dir = os.path.join(target_unpacked_dir, "dev")
    test_ted_dir = os.path.join(target_unpacked_dir, "test")

    prepare_dir(train_ted_dir)
    prepare_dir(val_ted_dir)
    prepare_dir(test_ted_dir)
    print('Creating manifests...')

    create_manifest(train_ted_dir, 'ted_train_manifest.csv', args.min_duration, args.max_duration)
    create_manifest(val_ted_dir, 'ted_val_manifest.csv')
    create_manifest(test_ted_dir, 'ted_test_manifest.csv')


if __name__ == "__main__":
    main()
