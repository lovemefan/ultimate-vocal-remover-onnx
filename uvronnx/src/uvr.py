# -*- coding:utf-8 -*-
# @FileName  :uvr.py
# @Time      :2023/8/2 10:47
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import os.path

import numpy as np
import librosa
from tqdm import tqdm

from uvronnx.src.config import UVR_CONFIG
from uvronnx.src.ortInferSession import UVROrtInferSession
from uvronnx.src.utils import spec_utils
from uvronnx.src.utils.AudioHelper import AudioReader
from uvronnx.src.utils.spec_utils import make_padding


class UVRModel:
    def __init__(self, model_path=None):
        project_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = model_path or os.path.join(project_dir, 'onnx/uvr-fp16-sim.onnx')
        assert os.path.exists(model_path), f"{model_path} is not exist"

        self.model = UVROrtInferSession({
            'model_path': model_path,
            'use_cuda': False
        })
        self.offset = 128
        self.window_size = 512

    def preprocess(x_spec):
        x_mag = np.abs(x_spec)
        x_phase = np.angle(x_spec)

        return x_mag, x_phase

    def separate_offline(self, mixed_audio, sample_rate=44100):
        if isinstance(mixed_audio, str):
            mixed_audio, sample_rate = AudioReader.read_wav_file(mixed_audio)

        x_wave, y_wave, x_spec_s, y_spec_s = {}, {}, {}, {}
        bands_n = len(UVR_CONFIG['band'])
        for d in range(bands_n, 0, -1):
            bp = UVR_CONFIG['band'][d]
            if d == bands_n:  # high-end band
                x_wave[d] = mixed_audio
                if x_wave[d].ndim == 1:
                    x_wave[d] = np.asfortranarray([x_wave[d], x_wave[d]])
            else:  # lower bands
                x_wave[d] = librosa.core.resample(x_wave[d + 1], orig_sr=UVR_CONFIG['band'][d + 1]['sr'], target_sr=bp['sr'],
                                                  res_type=bp['res_type'])
            # Stft of wave source
            x_spec_s[d] = spec_utils.wave_to_spectrogram_mt(x_wave[d], bp['hl'], bp['n_fft'], UVR_CONFIG['mid_side'],
                                                            UVR_CONFIG['mid_side_b2'], UVR_CONFIG['reverse'])
            # pdb.set_trace()
            if d == bands_n:
                input_high_end_h = (bp['n_fft'] // 2 - bp['crop_stop']) + (
                        UVR_CONFIG['pre_filter_stop'] - UVR_CONFIG['pre_filter_start'])
                input_high_end = x_spec_s[d][:, bp['n_fft'] // 2 - input_high_end_h:bp['n_fft'] // 2, :]

        x_spec_m = spec_utils.combine_spectrograms(x_spec_s, UVR_CONFIG)

        def preprocess(x_spec):
            x_mag = np.abs(x_spec).astype(np.float16)
            x_phase = np.angle(x_spec).astype(np.float16)
            return x_mag, x_phase

        x_mag, x_phase = preprocess(x_spec_m)

        coef = x_mag.max()
        x_mag_pre = x_mag / coef

        n_frame = x_mag_pre.shape[2]
        pad_l, pad_r, roi_size = make_padding(n_frame,
                                              self.window_size, self.offset)
        n_window = int(np.ceil(n_frame / roi_size))

        x_mag_pad = np.pad(
            x_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')

        preds = []

        iterations = [n_window]

        total_iterations = sum(iterations)
        for i in tqdm(range(n_window)):
            start = i * roi_size
            x_mag_window = x_mag_pad[None, :, :, start:start + self.window_size]
            # if (is_half == True): x_mag_window = x_mag_window.half()

            h = self.model(x_mag_window)
            pred = h[:, :, :, self.offset:-self.offset]
            assert pred.shape[3] > 0

            preds.append(pred[0])

        pred = np.concatenate(preds, axis=2)
        pred = pred[:, :, :n_frame]
        pred, x_mag, x_phase = pred * coef, x_mag, np.exp(1.j * x_phase)

        y_spec_m = pred * x_phase
        v_spec_m = x_spec_m - y_spec_m

        input_high_end_ = spec_utils.mirroring('mirroring', y_spec_m, input_high_end, UVR_CONFIG)
        wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, UVR_CONFIG, input_high_end_h,
                                                            input_high_end_)
        print('instruments done')

        input_high_end_ = spec_utils.mirroring('mirroring', v_spec_m, input_high_end, UVR_CONFIG)
        wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, UVR_CONFIG, input_high_end_h, input_high_end_)

        return wav_instrument, wav_vocals


if __name__ == '__main__':
    model = UVRModel()
    audio, sample_rate = AudioReader.read_wav_file('/Users/cenglingfan/Downloads/晴天.wav_-4key_fumin.wav')
    instrument, vocal = model.separate_offline(audio, sample_rate)
    print(instrument)
    print(vocal)