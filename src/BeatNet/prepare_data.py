# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>
# Data preparation script for BeatNet training.
# Extracts LOG_SPECT features and beat/downbeat annotations from raw audio,
# saves per-track pickle files ready for training.
#
# Usage:
#   python -m BeatNet.prepare_data --config configs/default.yaml \
#       --raw_dir /path/to/raw/datasets --dataset BALLROOM
#
# Expected raw directory structure:
#   {raw_dir}/{dataset_lower}/
#       audio/{split_or_genre}/{track}.wav
#       annotations/{track}.beats
#
# The .beats annotation format:
#   <time_in_seconds> <beat_number>
#   where beat_number == 1 means downbeat, anything else means regular beat.
#
# Output structure:
#   {data_dir}/{DATASET}/tracks/{DATASET}#{split}#{trackname}.pkl
#   {data_dir}/{DATASET}/tracks_list.pkl

import argparse
import os
import pickle
import sys
from collections import defaultdict

import librosa
import numpy as np
import yaml

from BeatNet.log_spect import LOG_SPECT


def parse_beats_file(label_path):
    """Parse a .beats annotation file into beat and downbeat time arrays."""
    beats = []
    downs = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            time_sec = float(parts[0])
            beat_num = int(parts[1])
            if beat_num == 1:
                downs.append(time_sec)
            else:
                beats.append(time_sec)
    return np.asarray(beats), np.asarray(downs)


def build_ground_truth(beats, downs, num_frames, sample_rate, hop_length):
    """Build a (3, num_frames) one-hot ground truth matrix: [beat, downbeat, non-beat]."""
    gt = np.zeros((3, num_frames), dtype=np.float32)

    if len(beats) > 0:
        beat_frames = librosa.time_to_frames(beats, sr=sample_rate, hop_length=hop_length)
        beat_frames = beat_frames[beat_frames < num_frames]
        gt[0, beat_frames] = 1

    if len(downs) > 0:
        down_frames = librosa.time_to_frames(downs, sr=sample_rate, hop_length=hop_length)
        down_frames = down_frames[down_frames < num_frames]
        gt[1, down_frames] = 1
        # Downbeat frames should NOT also be marked as beat in row 0
        gt[0, down_frames] = 0

    # Non-beat: frames where neither beat nor downbeat is active
    gt[2, np.sum(gt, axis=0) == 0] = 1

    assert int(np.sum(gt)) == num_frames, \
        f"Ground truth sum {int(np.sum(gt))} != num_frames {num_frames}"
    return gt


def discover_splits(audio_dir):
    """Discover splits (genre subdirectories) under an audio directory."""
    splits = []
    for entry in sorted(os.listdir(audio_dir)):
        full = os.path.join(audio_dir, entry)
        if os.path.isdir(full):
            splits.append(entry)
    return splits


def find_annotation(annotations_dir, track_name):
    """Find the .beats annotation file for a track, trying common naming patterns."""
    # Try exact match first
    for ext in ['.beats', '.beat']:
        path = os.path.join(annotations_dir, track_name + ext)
        if os.path.exists(path):
            return path
    # Try case-insensitive search
    for f in os.listdir(annotations_dir):
        base = os.path.splitext(f)[0]
        if base.lower() == track_name.lower() and f.endswith(('.beats', '.beat')):
            return os.path.join(annotations_dir, f)
    return None


def prepare_dataset(dataset_name, raw_dir, data_dir, feature_extractor, sample_rate, hop_length):
    """Prepare a single dataset: extract features, parse annotations, save pickles."""
    dataset_lower = dataset_name.lower()
    dataset_raw = os.path.join(raw_dir, dataset_lower)

    audio_dir = os.path.join(dataset_raw, 'audio')
    annotations_dir = os.path.join(dataset_raw, 'annotations')

    if not os.path.isdir(audio_dir):
        print(f"ERROR: Audio directory not found: {audio_dir}")
        sys.exit(1)
    if not os.path.isdir(annotations_dir):
        print(f"ERROR: Annotations directory not found: {annotations_dir}")
        sys.exit(1)

    # Output directories
    tracks_dir = os.path.join(data_dir, dataset_name, 'tracks')
    os.makedirs(tracks_dir, exist_ok=True)

    splits = discover_splits(audio_dir)
    if not splits:
        # No subdirectories — treat audio_dir itself as a single split
        splits = ['default']

    tracks_list = defaultdict(list)
    total_processed = 0
    total_skipped = 0

    for split in splits:
        if split == 'default':
            split_audio_dir = audio_dir
        else:
            split_audio_dir = os.path.join(audio_dir, split)

        wav_files = sorted([f for f in os.listdir(split_audio_dir) if f.endswith(('.wav', '.mp3', '.flac'))])
        print(f"  Split '{split}': {len(wav_files)} audio files")

        for wav_file in wav_files:
            track_name = os.path.splitext(wav_file)[0]
            track_id = f"{dataset_name}#{split}#{track_name}"
            wav_path = os.path.join(split_audio_dir, wav_file)

            # Find annotation
            label_path = find_annotation(annotations_dir, track_name)
            if label_path is None:
                print(f"    SKIP {track_name}: no annotation found")
                total_skipped += 1
                continue

            # Load audio
            wav, _ = librosa.load(wav_path, sr=sample_rate)

            # Extract features
            feats = feature_extractor.process_audio(wav)  # returns (272, T) after .T in LOG_SPECT
            num_frames = feats.shape[1]

            # Parse annotations
            beats, downs = parse_beats_file(label_path)

            # Build ground truth
            gt = build_ground_truth(beats, downs, num_frames, sample_rate, hop_length)

            # Filter: require at least 4 beats and 4 downbeats
            if gt[0].sum() + gt[1].sum() < 4 or gt[1].sum() < 4:
                print(f"    SKIP {track_name}: insufficient annotations "
                      f"(beats={int(gt[0].sum())}, downbeats={int(gt[1].sum())})")
                total_skipped += 1
                continue

            # Compute times
            frame_idcs = np.arange(num_frames)
            times = librosa.frames_to_time(frame_idcs, sr=sample_rate, hop_length=hop_length)

            # Save per-track pickle
            data = {
                'feats': feats.astype(np.float32),
                'times': times.astype(np.float32),
                'ground_truth': gt,
            }
            pkl_path = os.path.join(tracks_dir, track_id + '.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(data, f)

            tracks_list[split].append(track_id)
            total_processed += 1

    # Save tracks list manifest
    manifest_path = os.path.join(data_dir, dataset_name, 'tracks_list.pkl')
    with open(manifest_path, 'wb') as f:
        pickle.dump(dict(tracks_list), f)

    print(f"  Done: {total_processed} tracks processed, {total_skipped} skipped")
    print(f"  Manifest saved to: {manifest_path}")
    return tracks_list


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for BeatNet training')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--raw_dir', type=str, required=True,
                        help='Root directory containing raw dataset folders')
    parser.add_argument('--dataset', type=str, nargs='+', default=None,
                        help='Dataset name(s) to process (e.g., BALLROOM GTZAN). '
                             'If not specified, processes all datasets from config.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Output directory for prepared data (overrides config)')
    args = parser.parse_args()

    # Load config
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    sample_rate = config.get('sample_rate', 22050)
    hop_length = config.get('hop_length', 441)
    win_length = config.get('win_length', 1411)
    n_bands = config.get('n_bands', 24)
    data_dir = args.data_dir or config.get('data_dir', './data')

    # Determine which datasets to process
    if args.dataset:
        datasets = args.dataset
    else:
        ds_config = config.get('datasets', {})
        datasets = ds_config.get('train', []) + ds_config.get('test', [])

    if not datasets:
        print("ERROR: No datasets specified. Use --dataset or provide a config with datasets listed.")
        sys.exit(1)

    # Initialize feature extractor (same as inference)
    feature_extractor = LOG_SPECT(
        sample_rate=sample_rate,
        win_length=win_length,
        hop_size=hop_length,
        n_bands=[n_bands],
        mode='online'  # offline/online mode for full-file processing
    )

    print(f"Feature extractor: LOG_SPECT (dim={config.get('feature_dim', 272)})")
    print(f"Sample rate: {sample_rate}, hop: {hop_length}, win: {win_length}")
    print(f"Output directory: {data_dir}")
    print()

    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        prepare_dataset(dataset_name, args.raw_dir, data_dir, feature_extractor, sample_rate, hop_length)
        print()

    print("All done.")


if __name__ == '__main__':
    main()
