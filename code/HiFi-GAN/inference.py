# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import argparse
import glob
import json
import os
import random
import sys
import time
from multiprocessing import Manager, Pool
from pathlib import Path
import torch
import librosa
import numpy as np
import torch
from scipy.io.wavfile import write
import soundfile as sf
from dataset_f0recon import CodeDataset, parse_manifest, mel_spectrogram, \
    MAX_WAV_VALUE, load_audio
from dataset_f0recon import get_yaapt_f0
from utils import AttrDict
from models import CodeGenerator
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
from librosa.util import normalize


h = None
device = None

def stream(message):
    sys.stdout.write(f"\r{message}")


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def load_checkpoint(filepath):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location='cpu')
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def generate(h, generator, code):
    start = time.time()
    y_g_hat = generator(**code)
    if type(y_g_hat) is tuple:
        y_g_hat = y_g_hat[0]
    rtf = (time.time() - start) / (y_g_hat.shape[-1] / h.sampling_rate)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    return audio, rtf

def init_worker(queue, arguments):
    import logging
    logging.getLogger().handlers = []

    global generator
    global f0_stats
    global spkrs_emb
    global dataset
    global spkr_dataset
    global idx
    global device
    global a
    global h
    global spkrs

    a = arguments
    idx = queue.get()
    device = idx

    if os.path.isdir(a.checkpoint_file):
        config_file = os.path.join(a.checkpoint_file, 'config.json')
    else:
        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = CodeGenerator(h).to(idx)
    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    state_dict_g = load_checkpoint(cp_g)
    generator.load_state_dict(state_dict_g['generator'])


    if a.code_file is not None:
        dataset = [x.strip().split('|') for x in open(a.code_file).readlines()]

        def parse_code(c):
            c = [int(v) for v in c.split(" ")]
            return [torch.LongTensor(c).numpy()]

        dataset = [(parse_code(x[1]), None, x[0], None) for x in dataset]
    else:
        file_list = parse_manifest(a.input_code_file)
        dataset = CodeDataset(file_list, -1, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
                              h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device,
                              f0=h.get('f0', None), multispkr=h.get('multispkr', None),
                              f0_stats=h.get('f0_stats', None), f0_normalize=h.get('f0_normalize', False),
                              f0_feats=h.get('f0_feats', False), f0_median=h.get('f0_median', False),
                              f0_interp=h.get('f0_interp', False), vqvae=h.get('code_vq_params', False),
                              pad=a.pad, pitch_folder=a.pitch_folder, emo_folder=a.emo_folder)

    if a.unseen_f0:
        dataset.f0_stats = torch.load(a.unseen_f0)

    os.makedirs(a.output_dir, exist_ok=True)

    if h.get('multispkr', None):
        spkrs = random.sample(range(len(dataset.id_to_spkr)), k=min(5, len(dataset.id_to_spkr)))

    if a.f0_stats and h.get('f0', None) is not None:
        f0_stats = torch.load(a.f0_stats)

    generator.eval()
    generator.remove_weight_norm()

    # fix seed
    seed = 52 + idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def inference(item_index):
    speaker_id = {}

    for ind in range(11, 21):
        speaker_id["00"+str(ind)] = ind-11
    code, gt_audio, filename, _ = dataset[item_index]
    code = {k: torch.tensor(v).to(device).unsqueeze(0) for k, v in code.items()}

    if a.parts:
        parts = Path(filename).parts
        fname_out_name = '_'.join(parts[-3:])[:-4]
    else:
        fname_out_name = Path(filename).stem
   

    if int(fname_out_name[5:11]) < 350:
        if h.get('f0_vq_params', None) or h.get('f0_quantizer', None):
            to_remove = gt_audio.shape[-1] % (16 * 80)
            assert to_remove % h['code_hop_size'] == 0

            if to_remove != 0:
                to_remove_code = to_remove // h['code_hop_size']
                to_remove_f0 = to_remove // 320

                gt_audio = gt_audio[:-to_remove]
                code['code'] = code['code'][..., :-to_remove_code]
                code['f0'] = code['f0'][..., :-to_remove_f0]

        new_code = dict(code)
        if 'f0' in new_code:
            del new_code['f0']
            new_code['f0'] = code['f0']
        
        if h.get('multispkr', None) and a.convert:
            print("In conversion")
            reference_files = os.listdir("/folder/to/ESD/test/wavs")
            #Change line 194 for setting same/different source/reference speaker
            reference_files = [x for x in reference_files if x[:4] != fname_out_name[:4]]
            reference_files = [x for x in reference_files if int(x[5:11]) >= 350]
            reference_files = [x for x in reference_files if ".wav" in x]
            source_num = int(fname_out_name[5:11])
            #Change line 199 for setting same/different source/reference utterance
            reference_files = [x for x in reference_files if (int(x[5:11])-source_num)%350!=0]
            
            for i, filename in enumerate(reference_files):
                print(i, filename)
                emo_embed = np.load("/ZEST/code/F0_predictor/wav2vec_feats/" + filename.replace(".wav", ".npy"))
                feats = {}
                f0 = np.load("/ZEST/code/F0_predictor/pred_DSDT_f0" + fname_out_name + filename.replace(".wav", ".npy"))
                f0 = f0.astype(np.float32)
                trg_f0 = f0
                new_f0 = torch.tensor(f0)
                new_f0 = new_f0.squeeze(-1)
                code['f0'] = torch.FloatTensor(new_f0).to(device)
                code['f0'] = code['f0'].unsqueeze(0).unsqueeze(0)                
                if code['f0'].shape[-1] > new_code['f0'].shape[-1]:
                    code['f0'] = code['f0'][:, :, :new_code['f0'].shape[-1]]
                code["emo_embed"] = torch.tensor(emo_embed).unsqueeze(0).to(device)
                audio, rtf = generate(h, generator, code)

                output_file = os.path.join(a.output_dir, fname_out_name + filename)
                audio = librosa.util.normalize(audio.astype(np.float32))
                write(output_file, h.sampling_rate, audio)

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--code_file', default=None)
    parser.add_argument('--input_code_file', default="/ZEST/code/test_esd.txt")
    parser.add_argument('--output_dir', default='DSDT')
    parser.add_argument('--emo_folder', default='')
    parser.add_argument('--pitch_folder', default='')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--f0-stats', type=Path)
    parser.add_argument('--vc', action='store_true')
    parser.add_argument('--convert', action='store_true')
    parser.add_argument('--random-speakers', action='store_true')
    parser.add_argument('--pad', default=None, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--parts', action='store_true')
    parser.add_argument('--unseen-f0', type=Path)
    parser.add_argument('-n', type=int, default=1500)
    a = parser.parse_args()

    seed = 52
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ids = list(range(1))
    manager = Manager()
    idQueue = manager.Queue()
    for i in ids:
        idQueue.put(i)

    if os.path.isdir(a.checkpoint_file):
        config_file = os.path.join(a.checkpoint_file, 'config.json')
    else:
        config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    if os.path.isdir(a.checkpoint_file):
        cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
    else:
        cp_g = a.checkpoint_file
    if not os.path.isfile(cp_g) or not os.path.exists(cp_g):
        print(f"Didn't find checkpoints for {cp_g}")
        return

    if a.code_file is not None:
        dataset = [x.strip().split('|') for x in open(a.code_file).readlines()]

        def parse_code(c):
            c = [int(v) for v in c.split(" ")]
            return [torch.LongTensor(c).numpy()]

        dataset = [(parse_code(x[1]), None, x[0], None) for x in dataset]
    else:
        file_list = parse_manifest(a.input_code_file)
        dataset = CodeDataset(file_list, -1, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size, h.win_size,
                              h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss, device=device,
                              f0=h.get('f0', None), multispkr=h.get('multispkr', None),
                              f0_stats=h.get('f0_stats', None), f0_normalize=h.get('f0_normalize', False),
                              f0_feats=h.get('f0_feats', False), f0_median=h.get('f0_median', False),
                              f0_interp=h.get('f0_interp', False), vqvae=h.get('code_vq_params', False),
                              pad=a.pad, pitch_folder=a.pitch_folder, emo_folder=a.emo_folder)

    if a.debug:
        ids = list(range(1))
        import queue
        idQueue = queue.Queue()
        for i in ids:
            idQueue.put(i)
        init_worker(idQueue, a)

        for i in range(0, len(dataset)):
            inference(i)
            bar = progbar(i, len(dataset))
            message = f'{bar} {i}/{len(dataset)} '
            stream(message)
            if a.n != -1 and i > a.n:
                break
    else:
        idx = list(range(len(dataset)))
        random.shuffle(idx)
        with Pool(1, init_worker, (idQueue, a)) as pool:
            for i, _ in enumerate(pool.imap(inference, idx), 1):
                bar = progbar(i, len(idx))
                message = f'{bar} {i}/{len(idx)} '
                stream(message)
                if a.n != -1 and i > a.n:
                    break


if __name__ == '__main__':
    main()
