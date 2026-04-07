# Copyright (c) 2025-2026 Hannes Göök
# MIT License - GDForge
# https://github.com/hannesgook/gdforge

from typing import Tuple
import numpy as np
import librosa
from scipy.signal import find_peaks
from settings import PeakSettings

def load_audio_mono(path: str, sr: int) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr

def compute_env(y: np.ndarray, sr: int, ps: PeakSettings) -> Tuple[np.ndarray, np.ndarray]:
    if ps.use_onset_env:
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=ps.hop, aggregate=np.median)
    else:
        env = librosa.feature.rms(y=y, frame_length=ps.frame, hop_length=ps.hop, center=False)[0]
    env = np.asarray(env, dtype=np.float32)
    t_env = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=ps.hop).astype(np.float64)
    return env, t_env

def detect_peaks_from_env(env: np.ndarray, sr: int, ps: PeakSettings) -> Tuple[np.ndarray, float]:
    thr_dyn = float(np.percentile(env, ps.peak_percentile))
    height = max(thr_dyn, float(ps.peak_abs))
    min_dist = max(1, int(round(ps.min_sep_s * sr / ps.hop)))
    idx, _ = find_peaks(env, height=height, distance=min_dist)
    times = librosa.frames_to_time(idx, sr=sr, hop_length=ps.hop).astype(np.float64)
    return times, height

def analyze_audio(audio_path: str, ps: PeakSettings) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, float]:
    y, sr = load_audio_mono(audio_path, ps.sr)
    env, t_env = compute_env(y, sr, ps)
    peak_times, thr = detect_peaks_from_env(env, sr, ps)

    return y, sr, env, t_env, peak_times, thr
