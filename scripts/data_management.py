from pathlib import Path

import numpy as np

def get_observation_nums(data_dir: Path) -> np.ndarray:
    return np.array([int(fn.stem.split("_")[0]) for fn in data_dir.glob("*_stacked_spectrograms.npy")])

def get_stratified_obs_nums(data_dir: Path) -> dict:
    stratified_obs_nums = dict()
    for subset in ("train", "test", "validation"):
        stratified_obs_nums[subset] = get_observation_nums(data_dir / subset)
    return stratified_obs_nums

def make_dataset_name(nfft, ts_crop_width, vr_crop_width):
    return f"data_fft-{nfft}_tscropwidth-{abs(ts_crop_width[0])}-{ts_crop_width[1]}_vrcropwidth-{abs(vr_crop_width[0])}-{vr_crop_width[1]}"

def parse_dataset_name(dataset_name):
    components = dataset_name.split("_")
    
    fft = int(components[1].split("-")[1])

    ts_crop_width = components[2].split("-")[1:]
    ts_crop_width = (-int(ts_crop_width[0]), int(ts_crop_width[1]))
    
    vr_crop_width = components[3].split("-")[1:]
    vr_crop_width = (-int(vr_crop_width[0]), int(vr_crop_width[1]))
    
    return fft, ts_crop_width, vr_crop_width
