import os
import time
import numpy as np
import argparse
from speech_tools import *
from multiprocessing import Pool
from hparams import *


def process(folder):
    save_folder = os.path.join(argv.output_dir, os.path.basename(folder))
    print('save folder', save_folder)
    os.makedirs(save_folder, exist_ok=True)

    X = []
    for file in glob.glob(folder + '/*.wav'):
        wav, _ = librosa.load(file, sr=sampling_rate, mono=True)
        wav *= 1. / max(0.01, np.max(np.abs(wav)))

        # Silence Removal
        wav_splitted = librosa.effects.split(wav, top_db=48)

        for s in range(wav_splitted.shape[0]):
            x = wav[wav_splitted[s][0]:wav_splitted[s][1]]
            X = np.concatenate([X, x], axis=0)
    X *= 1. / max(0.01, np.max(np.abs(X)))
    wavlen = X.shape[0]
    crop_size = wavlen // argv.divs
    start = 0

    for i in range(argv.divs):
        if (i == argv.divs - 1):
            sub = X[start:]
        else:
            sub = X[start:start + crop_size]

        start += crop_size
        sub = sub.astype(np.float32)
        librosa.output.write_wav(
            os.path.join(save_folder, "{}_".format(i) + os.path.basename(folder) + ".wav"), sub,
            sampling_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess wav files to train RelGAN')
    parser.add_argument('--dataset_dir', type=str, help='Directory for preprocess', default="datasets")
    parser.add_argument('--output_dir', type=str, help='Directory for save output', default="datasets_splitted")
    parser.add_argument('--parallel', type=bool, help='Boolean of parallel', default=True)
    parser.add_argument('--divs', type=str, help='Batch size', default=64)
    argv = parser.parse_args()
    print('args ', argv)

    os.makedirs(argv.output_dir, exist_ok=True)

    folders = glob.glob(argv.dataset_dir + "/*")
    TIME = time.time()
    cores = min(len(folders), 4)
    if argv.parallel:
        p = Pool(cores)
        p.map(process, folders)

        p.close()
    else:
        for f in folders:
            process(f, argv)

    print(time.time() - TIME)
