import argparse
from pathlib import Path

import numpy as np
import torch

from soundnet import SoundNet
from util import gen_audio_from_dir

# terminal command for mp4 -> mp3 
# for file in /shared/youngkim/dataset/videos/*;do filename=$(basename $file .mp4); ffmpeg -y -i $file -ac 1 -f mp3 -ar 22050 /shared/youngkim/dataset/mp3/${filename}.mp3; done

# Script global configs
LEN_WAVEFORM = 22050 * 20

local_config = {
	'batch_size': 1,
	'eps': 1e-5,
	'sample_rate': 22050,
	'load_size': 22050 * 20,
	'name_scope': 'SoundNet_TF',
	'phase': 'extract',
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default='soundnet/models/sound8.npy',
                        help='Path to the .npy file with the SoundNet weights')

    parser.add_argument('-i', '--input_dir', type=str, default='/shared/youngkim/dataset/mp3',
                        help='Directory with the audio files to extract SoundNet feats')

    parser.add_argument('-o', '--output_dir', type=str, default='/shared/youngkim/dataset/raw',
                        help='Dir where the audio features will be stored')
    
    parser.add_argument('-f', '--file_ext', type=str, default='.mp3',
                        help='File extension of the audio files')

    return parser.parse_args()


def extract_features(args):
    
    model = SoundNet()
    model.load_weights(args.model_path)
    model.eval()
    
    error_count = 0
    for sound_sample, audio_path in gen_audio_from_dir(args.input_dir,
                                                       config=local_config,
                                                       file_ext=args.file_ext):
        if sound_sample is None:
            error_count += 1
            continue
        print(audio_path, sound_sample.shape)
        new_sample = torch.from_numpy(sound_sample)
        feats = model.forward(new_sample)

        output_path = Path(args.output_dir, f'{Path(audio_path).stem}.npy')
        print(output_path, feats.keys())
        np.save(output_path, feats)

    if error_count > 0:
        print(f'Could not process {error_count} audio files correctly.')


if __name__ == '__main__':
    args = parse_args()
    extract_features(args)
