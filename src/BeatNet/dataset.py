# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>
# PyTorch Dataset for BeatNet training.

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class BeatNetDataset(Dataset):
    """PyTorch Dataset for BeatNet training/validation/testing.

    Each item is a dict with keys: 'feats', 'times', 'ground_truth'.
    For training, a random contiguous window of seq_len frames is cropped.
    For validation/testing (seq_len=None), the full track is returned.
    """

    def __init__(self, track_ids, tracks_dirs, seq_len=None, seed=42):
        """
        Parameters
        ----------
        track_ids : list of str
            Track identifiers, e.g. "BALLROOM#ChaChaCha#track001"
        tracks_dirs : dict
            Mapping from dataset name to the directory containing its track pickles.
            e.g. {"BALLROOM": "/data/BALLROOM/tracks"}
        seq_len : int or None
            Number of frames to crop per sample. None = return full track.
        seed : int
            Random seed for reproducible cropping.
        """
        self.track_ids = track_ids
        self.tracks_dirs = tracks_dirs
        self.seq_len = seq_len
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        dataset_name = track_id.split('#')[0]
        tracks_dir = self.tracks_dirs[dataset_name]

        pkl_path = os.path.join(tracks_dir, track_id + '.pkl')
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        feats = data['feats']           # (272, T)
        times = data['times']           # (T,)
        ground_truth = data['ground_truth']  # (3, T)

        if self.seq_len is not None:
            num_frames = feats.shape[-1]
            if num_frames <= self.seq_len:
                # Pad if track is shorter than seq_len
                pad_len = self.seq_len - num_frames
                feats = np.pad(feats, ((0, 0), (0, pad_len)), mode='constant')
                times = np.pad(times, (0, pad_len), mode='constant')
                ground_truth = np.pad(ground_truth, ((0, 0), (0, pad_len)), mode='constant')
                # Mark padded frames as non-beat
                ground_truth[2, num_frames:] = 1
            else:
                start = self.rng.randint(0, num_frames - self.seq_len)
                end = start + self.seq_len
                feats = feats[..., start:end]
                times = times[start:end]
                ground_truth = ground_truth[..., start:end]

        return {
            'feats': torch.from_numpy(feats.copy()),
            'times': torch.from_numpy(times.copy()),
            'ground_truth': torch.from_numpy(ground_truth.copy()),
        }


def build_datasets(config):
    """Build train, validation, and test datasets from prepared data.

    Parameters
    ----------
    config : dict
        Training configuration (loaded from YAML).

    Returns
    -------
    train_dataset, val_dataset, test_dataset : BeatNetDataset
    """
    data_dir = config['data_dir']
    ds_config = config['datasets']
    train_datasets = ds_config.get('train', [])
    test_datasets = ds_config.get('test', [])
    dataset_weights = config.get('dataset_weights', {})
    train_val_split = config.get('train_val_split', 0.9)
    seq_len = config.get('seq_len', 400)
    seed = config.get('seed', 42)

    rng = np.random.RandomState(seed)

    train_ids = []
    val_ids = []
    test_ids = []
    tracks_dirs = {}

    # Process training datasets
    for ds_name in train_datasets:
        ds_dir = os.path.join(data_dir, ds_name)
        tracks_dir = os.path.join(ds_dir, 'tracks')
        tracks_dirs[ds_name] = tracks_dir

        manifest_path = os.path.join(ds_dir, 'tracks_list.pkl')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Run prepare_data.py first to extract features for {ds_name}."
            )

        with open(manifest_path, 'rb') as f:
            tracks_list = pickle.load(f)

        # Collect all track IDs from all splits
        all_ids = []
        for split_ids in tracks_list.values():
            all_ids.extend(split_ids)

        # Shuffle and split into train/val
        rng.shuffle(all_ids)
        n_train = int(len(all_ids) * train_val_split)
        split_train = all_ids[:n_train]
        split_val = all_ids[n_train:]

        # Apply oversampling weight
        weight = dataset_weights.get(ds_name, 1)
        for _ in range(weight):
            train_ids.extend(split_train)
        val_ids.extend(split_val)

    # Process test datasets
    for ds_name in test_datasets:
        ds_dir = os.path.join(data_dir, ds_name)
        tracks_dir = os.path.join(ds_dir, 'tracks')
        tracks_dirs[ds_name] = tracks_dir

        manifest_path = os.path.join(ds_dir, 'tracks_list.pkl')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Run prepare_data.py first to extract features for {ds_name}."
            )

        with open(manifest_path, 'rb') as f:
            tracks_list = pickle.load(f)

        for split_ids in tracks_list.values():
            test_ids.extend(split_ids)

    print(f"Dataset splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    train_dataset = BeatNetDataset(train_ids, tracks_dirs, seq_len=seq_len, seed=seed)
    val_dataset = BeatNetDataset(val_ids, tracks_dirs, seq_len=None, seed=seed)
    test_dataset = BeatNetDataset(test_ids, tracks_dirs, seq_len=None, seed=seed)

    return train_dataset, val_dataset, test_dataset
