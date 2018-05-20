import os
import wget
import tarfile
import argparse
import subprocess
from utils import create_manifest
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser(description='Processes and downloads LibriSpeech dataset.')
parser.add_argument("--target-dir", default='LibriSpeech_dataset/', type=str, help="Directory to store the dataset.")
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--files-to-use', default="train-clean-100.tar.gz,"
                                              "train-clean-360.tar.gz,train-other-500.tar.gz,"
                                              "dev-clean.tar.gz,dev-other.tar.gz,"
                                              "test-clean.tar.gz,test-other.tar.gz", type=str,
                    help='list of file names to download')
parser.add_argument('--min-duration', default=1, type=int,
                    help='Prunes training samples shorter than the min duration (given in seconds, default 1)')
parser.add_argument('--max-duration', default=15, type=int,
                    help='Prunes training samples longer than the max duration (given in seconds, default 15)')
args = parser.parse_args()

LIBRI_SPEECH_URLS = {
    "train": ["http://www.openslr.org/resources/12/train-clean-100.tar.gz",
              "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
              "http://www.openslr.org/resources/12/train-other-500.tar.gz"],

    "val": ["http://www.openslr.org/resources/12/dev-clean.tar.gz",
            "http://www.openslr.org/resources/12/dev-other.tar.gz"],

    "test_clean": ["http://www.openslr.org/resources/12/test-clean.tar.gz"],
    "test_other": ["http://www.openslr.org/resources/12/test-other.tar.gz"]
}


def _preprocess_transcript(phrase):
    return phrase.strip().upper()


def _process_file(wav_dir, txt_dir, base_filename, root_dir):
    full_recording_path = os.path.join(root_dir, base_filename)
    assert os.path.exists(full_recording_path) and os.path.exists(root_dir)
    wav_recording_path = os.path.join(wav_dir, base_filename.replace(".flac", ".wav"))
    subprocess.call(["sox {}  -r {} -b 16 -c 1 {}".format(full_recording_path, str(args.sample_rate),
                                                          wav_recording_path)], shell=True)
    # process transcript
    txt_transcript_path = os.path.join(txt_dir, base_filename.replace(".flac", ".txt"))
    transcript_file = os.path.join(root_dir, "-".join(base_filename.split('-')[:-1]) + ".trans.txt")
    assert os.path.exists(transcript_file), "Transcript file {} does not exist.".format(transcript_file)
    transcriptions = open(transcript_file).read().strip().split("\n")
    transcriptions = {t.split()[0].split("-")[-1]: " ".join(t.split()[1:]) for t in transcriptions}
    with open(txt_transcript_path, "w") as f:
        key = base_filename.replace(".flac", "").split("-")[-1]
        assert key in transcriptions, "{} is not in the transcriptions".format(key)
        f.write(_preprocess_transcript(transcriptions[key]))
        f.flush()


def main():
    target_dl_dir = args.target_dir
    if not os.path.exists(target_dl_dir):
        os.makedirs(target_dl_dir)
    files_to_dl = args.files_to_use.strip().split(',')
    for split_type, lst_libri_urls in LIBRI_SPEECH_URLS.items():
        split_dir = os.path.join(target_dl_dir, split_type)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        split_wav_dir = os.path.join(split_dir, "wav")
        if not os.path.exists(split_wav_dir):
            os.makedirs(split_wav_dir)
        split_txt_dir = os.path.join(split_dir, "txt")
        if not os.path.exists(split_txt_dir):
            os.makedirs(split_txt_dir)
        extracted_dir = os.path.join(split_dir, "LibriSpeech")
        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)
        for url in lst_libri_urls:
            # check if we want to dl this file
            dl_flag = False
            for f in files_to_dl:
                if url.find(f) != -1:
                    dl_flag = True
            if not dl_flag:
                print("Skipping url: {}".format(url))
                continue
            filename = url.split("/")[-1]
            target_filename = os.path.join(split_dir, filename)
            if not os.path.exists(target_filename):
                wget.download(url, split_dir)
            print("Unpacking {}...".format(filename))
            tar = tarfile.open(target_filename)
            tar.extractall(split_dir)
            tar.close()
            os.remove(target_filename)
            print("Converting flac files to wav and extracting transcripts...")
            assert os.path.exists(extracted_dir), "Archive {} was not properly uncompressed.".format(filename)
            for root, subdirs, files in tqdm(os.walk(extracted_dir)):
                for f in files:
                    if f.find(".flac") != -1:
                        _process_file(wav_dir=split_wav_dir, txt_dir=split_txt_dir,
                                      base_filename=f, root_dir=root)

            print("Finished {}".format(url))
            shutil.rmtree(extracted_dir)
        if split_type == 'train':  # Prune to min/max duration
            create_manifest(split_dir, 'libri_' + split_type + '_manifest.csv', args.min_duration, args.max_duration)
        else:
            create_manifest(split_dir, 'libri_' + split_type + '_manifest.csv')


if __name__ == "__main__":
    main()
