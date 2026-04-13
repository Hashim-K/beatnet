"""
Test training pipeline with synthetic toy data.

Generates fake audio tracks with known beat/downbeat positions,
runs data preparation, dataset loading, a short training loop,
and validation — verifying the full pipeline end-to-end.

Usage:
    python -m pytest test/test_training.py -v
    # or directly:
    python test/test_training.py
"""

import os
import pickle
import shutil
import sys
import tempfile

import numpy as np
import torch

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from BeatNet.model import BDA
from BeatNet.dataset import BeatNetDataset, build_datasets
from BeatNet.prepare_data import build_ground_truth, parse_beats_file
from BeatNet.train import load_config, validate

SAMPLE_RATE = 22050
HOP_LENGTH = 441  # 50 fps
FEATURE_DIM = 272


def make_toy_track(num_seconds=10, bpm=120, meter=4):
    """Generate a synthetic track with known beats/downbeats.

    Returns a dict matching the per-track pickle format:
        feats: (272, T), times: (T,), ground_truth: (3, T)
    """
    num_frames = int(num_seconds * SAMPLE_RATE / HOP_LENGTH)
    feats = np.random.randn(FEATURE_DIM, num_frames).astype(np.float32)
    times = np.arange(num_frames) * HOP_LENGTH / SAMPLE_RATE

    # Place beats every beat_interval seconds
    beat_interval = 60.0 / bpm
    beat_times = np.arange(0, num_seconds, beat_interval)
    # Every `meter`-th beat is a downbeat
    down_times = beat_times[::meter]
    beat_only_times = np.array([t for t in beat_times if t not in down_times])

    gt = build_ground_truth(beat_only_times, down_times, num_frames, SAMPLE_RATE, HOP_LENGTH)

    return {
        'feats': feats,
        'times': times.astype(np.float32),
        'ground_truth': gt,
    }


def make_toy_dataset(data_dir, dataset_name, num_tracks=6, num_seconds=10):
    """Create a toy prepared dataset on disk (pickles + manifest)."""
    ds_dir = os.path.join(data_dir, dataset_name)
    tracks_dir = os.path.join(ds_dir, 'tracks')
    os.makedirs(tracks_dir, exist_ok=True)

    tracks_list = {'default': []}
    for i in range(num_tracks):
        bpm = np.random.randint(80, 180)
        meter = np.random.choice([3, 4])
        track_id = f"{dataset_name}#default#track_{i:03d}"
        data = make_toy_track(num_seconds=num_seconds, bpm=bpm, meter=meter)

        pkl_path = os.path.join(tracks_dir, track_id + '.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        tracks_list['default'].append(track_id)

    manifest_path = os.path.join(ds_dir, 'tracks_list.pkl')
    with open(manifest_path, 'wb') as f:
        pickle.dump(tracks_list, f)

    return tracks_list


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_ground_truth():
    """Ground truth matrix has correct shape and is mutually exclusive."""
    beats = np.array([0.5, 1.0, 1.5, 2.5, 3.0, 3.5])
    downs = np.array([0.0, 2.0, 4.0])
    num_frames = 250  # ~5 seconds at 50fps

    gt = build_ground_truth(beats, downs, num_frames, SAMPLE_RATE, HOP_LENGTH)

    assert gt.shape == (3, num_frames)
    # Every frame is exactly one class
    assert np.allclose(gt.sum(axis=0), 1.0)
    # At least some beats and downbeats
    assert gt[0].sum() > 0, "No beats found"
    assert gt[1].sum() > 0, "No downbeats found"
    assert gt[2].sum() > 0, "No non-beat frames found"
    print("  test_build_ground_truth PASSED")


def test_parse_beats_file():
    """Parse a .beats annotation file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.beats', delete=False) as f:
        f.write("0.5 1\n1.0 2\n1.5 3\n2.0 1\n2.5 2\n3.0 3\n")
        f.flush()
        beats, downs = parse_beats_file(f.name)

    os.unlink(f.name)
    assert len(downs) == 2  # beat_number == 1
    assert len(beats) == 4  # beat_number != 1
    np.testing.assert_allclose(downs, [0.5, 2.0])
    print("  test_parse_beats_file PASSED")


def test_dataset_training_mode():
    """BeatNetDataset returns correct shapes in training mode (fixed seq_len)."""
    tmpdir = tempfile.mkdtemp()
    try:
        make_toy_dataset(tmpdir, 'TOY', num_tracks=4, num_seconds=10)

        manifest_path = os.path.join(tmpdir, 'TOY', 'tracks_list.pkl')
        with open(manifest_path, 'rb') as f:
            tracks_list = pickle.load(f)
        all_ids = tracks_list['default']

        tracks_dirs = {'TOY': os.path.join(tmpdir, 'TOY', 'tracks')}
        ds = BeatNetDataset(all_ids, tracks_dirs, seq_len=200, seed=0)

        assert len(ds) == 4
        sample = ds[0]
        assert sample['feats'].shape == (FEATURE_DIM, 200)
        assert sample['ground_truth'].shape == (3, 200)
        assert sample['times'].shape == (200,)
        # Ground truth still mutually exclusive after cropping
        assert torch.allclose(sample['ground_truth'].sum(dim=0),
                              torch.ones(200, dtype=torch.float32))
        print("  test_dataset_training_mode PASSED")
    finally:
        shutil.rmtree(tmpdir)


def test_dataset_validation_mode():
    """BeatNetDataset returns full tracks when seq_len=None."""
    tmpdir = tempfile.mkdtemp()
    try:
        make_toy_dataset(tmpdir, 'TOY', num_tracks=2, num_seconds=8)

        manifest_path = os.path.join(tmpdir, 'TOY', 'tracks_list.pkl')
        with open(manifest_path, 'rb') as f:
            tracks_list = pickle.load(f)
        all_ids = tracks_list['default']

        tracks_dirs = {'TOY': os.path.join(tmpdir, 'TOY', 'tracks')}
        ds = BeatNetDataset(all_ids, tracks_dirs, seq_len=None, seed=0)

        sample = ds[0]
        T = sample['feats'].shape[1]
        assert T > 0
        assert sample['ground_truth'].shape == (3, T)
        print("  test_dataset_validation_mode PASSED")
    finally:
        shutil.rmtree(tmpdir)


def test_model_train_forward():
    """train_forward produces correct output shape and is stateless."""
    model = BDA(FEATURE_DIM, 150, 2, 'cpu')

    x = torch.randn(4, 200, FEATURE_DIM)
    out = model.train_forward(x)
    assert out.shape == (4, 3, 200), f"Expected (4, 3, 200), got {out.shape}"

    # Stateless: calling twice with same input gives same result
    model.eval()
    with torch.no_grad():
        out1 = model.train_forward(x)
        out2 = model.train_forward(x)
    assert torch.allclose(out1, out2, atol=1e-6), "train_forward should be stateless"
    print("  test_model_train_forward PASSED")


def test_training_loop():
    """Run a short training loop on toy data and verify loss decreases."""
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    tmpdir = tempfile.mkdtemp()
    try:
        make_toy_dataset(tmpdir, 'TOY_TRAIN', num_tracks=8, num_seconds=10)

        manifest_path = os.path.join(tmpdir, 'TOY_TRAIN', 'tracks_list.pkl')
        with open(manifest_path, 'rb') as f:
            tracks_list = pickle.load(f)
        all_ids = tracks_list['default']

        tracks_dirs = {'TOY_TRAIN': os.path.join(tmpdir, 'TOY_TRAIN', 'tracks')}
        ds = BeatNetDataset(all_ids, tracks_dirs, seq_len=200, seed=0)
        loader = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)

        model = BDA(FEATURE_DIM, 150, 2, 'cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        class_weights = torch.FloatTensor([50, 400, 5])

        losses = []
        model.train()
        for epoch in range(5):
            epoch_loss = []
            for batch in loader:
                optimizer.zero_grad()
                feats = batch['feats'].transpose(1, 2)  # (B, T, 272)
                gt = batch['ground_truth']                # (B, 3, T)
                preds = model.train_forward(feats)        # (B, 3, T)
                targets = torch.argmax(gt, dim=1)         # (B, T)
                loss = F.cross_entropy(preds, targets, weight=class_weights)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            losses.append(np.mean(epoch_loss))

        # Loss should generally decrease (or at least not blow up)
        assert losses[-1] < losses[0] * 2, \
            f"Loss did not behave reasonably: {losses[0]:.4f} -> {losses[-1]:.4f}"
        assert all(np.isfinite(losses)), "Loss contains NaN/Inf"
        print(f"  test_training_loop PASSED (loss: {losses[0]:.4f} -> {losses[-1]:.4f})")
    finally:
        shutil.rmtree(tmpdir)


def test_validation():
    """Run validation on toy data and verify it returns F-measure scores."""
    from torch.utils.data import DataLoader

    tmpdir = tempfile.mkdtemp()
    try:
        # Create toy tracks with very clear beat patterns
        make_toy_dataset(tmpdir, 'TOY_VAL', num_tracks=3, num_seconds=15)

        manifest_path = os.path.join(tmpdir, 'TOY_VAL', 'tracks_list.pkl')
        with open(manifest_path, 'rb') as f:
            tracks_list = pickle.load(f)
        all_ids = tracks_list['default']

        tracks_dirs = {'TOY_VAL': os.path.join(tmpdir, 'TOY_VAL', 'tracks')}
        ds = BeatNetDataset(all_ids, tracks_dirs, seq_len=None, seed=0)
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        model = BDA(FEATURE_DIM, 150, 2, 'cpu')
        model.eval()

        # With a random untrained model, F-measures will be low but should not crash
        beat_f, down_f = validate(model, loader, 'DBN', 'cpu')

        assert isinstance(beat_f, float)
        assert isinstance(down_f, float)
        assert 0.0 <= beat_f <= 1.0
        assert 0.0 <= down_f <= 1.0
        print(f"  test_validation PASSED (beat_F={beat_f:.4f}, down_F={down_f:.4f})")
    finally:
        shutil.rmtree(tmpdir)


def test_weight_compatibility():
    """Weights saved during training can be loaded by inference model."""
    model = BDA(FEATURE_DIM, 150, 2, 'cpu')

    # Simulate saving weights as training does
    tmpdir = tempfile.mkdtemp()
    try:
        weights_path = os.path.join(tmpdir, 'model_weights.pt')
        torch.save(model.state_dict(), weights_path)

        # Load into a fresh model (as inference code does)
        model2 = BDA(FEATURE_DIM, 150, 2, 'cpu')
        model2.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)

        # Verify outputs match
        model.eval()
        model2.eval()
        x = torch.randn(1, 50, FEATURE_DIM)
        with torch.no_grad():
            out1 = model.train_forward(x)
            out2 = model2.train_forward(x)
        assert torch.allclose(out1, out2, atol=1e-6)
        print("  test_weight_compatibility PASSED")
    finally:
        shutil.rmtree(tmpdir)


def test_full_pipeline():
    """End-to-end: create data, build datasets, train, validate, save/load weights."""
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    tmpdir = tempfile.mkdtemp()
    try:
        # 1. Create toy datasets
        make_toy_dataset(tmpdir, 'TRAIN_DS', num_tracks=8, num_seconds=10)
        make_toy_dataset(tmpdir, 'TEST_DS', num_tracks=3, num_seconds=12)

        # 2. Build datasets via config
        config = {
            'data_dir': tmpdir,
            'datasets': {'train': ['TRAIN_DS'], 'test': ['TEST_DS']},
            'dataset_weights': {'TRAIN_DS': 2},
            'train_val_split': 0.75,
            'seq_len': 200,
            'seed': 42,
        }
        train_ds, val_ds, test_ds = build_datasets(config)
        assert len(train_ds) > 0, "Empty train set"
        assert len(val_ds) > 0, "Empty val set"
        assert len(test_ds) > 0, "Empty test set"

        # 3. Train for a few steps
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        model = BDA(FEATURE_DIM, 150, 2, 'cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        class_weights = torch.FloatTensor([50, 400, 5])

        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            feats = batch['feats'].transpose(1, 2)
            preds = model.train_forward(feats)
            targets = torch.argmax(batch['ground_truth'], dim=1)
            loss = F.cross_entropy(preds, targets, weight=class_weights)
            loss.backward()
            optimizer.step()
            break  # just one batch

        # 4. Validate
        beat_f, down_f = validate(model, val_loader, 'DBN', 'cpu')
        assert isinstance(beat_f, float)

        # 5. Save and reload weights
        weights_path = os.path.join(tmpdir, 'test_weights.pt')
        torch.save(model.state_dict(), weights_path)

        model2 = BDA(FEATURE_DIM, 150, 2, 'cpu')
        model2.load_state_dict(torch.load(weights_path, map_location='cpu'), strict=False)

        print(f"  test_full_pipeline PASSED (train={len(train_ds)}, val={len(val_ds)}, "
              f"test={len(test_ds)}, beat_F={beat_f:.4f})")
    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_build_ground_truth,
    test_parse_beats_file,
    test_dataset_training_mode,
    test_dataset_validation_mode,
    test_model_train_forward,
    test_training_loop,
    test_validation,
    test_weight_compatibility,
    test_full_pipeline,
]


if __name__ == '__main__':
    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    if failed > 0:
        sys.exit(1)
