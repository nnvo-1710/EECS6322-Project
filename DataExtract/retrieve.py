import h5py
import numpy as np
import os
import glob
from sklearn.utils import shuffle

# --- Normalization Methods ---
def normalize_per_channel_dataset(clips):
    """Normalize each EEG channel independently across the dataset."""
    mean = np.mean(clips, axis=(0, 2, 3), keepdims=True)  # Shape: (1, 19, 1, 1)
    std = np.std(clips, axis=(0, 2, 3), keepdims=True)
    return (clips - mean) / (std + 1e-8)

def normalize_global_dataset(clips):
    """Normalize all EEG values using the global dataset mean/std."""
    mean = np.mean(clips)
    std = np.std(clips)
    return (clips - mean) / (std + 1e-8)

def normalize_per_clip_channel(clips):
    """Normalize each clip's channels independently."""
    mean = np.mean(clips, axis=(2, 3), keepdims=True)  # Shape: (batch_size, 19, 1, 1)
    std = np.std(clips, axis=(2, 3), keepdims=True)
    return (clips - mean) / (std + 1e-8)

def normalize_per_clip_whole(clips):
    """Normalize each clip using its overall mean/std."""
    mean = np.mean(clips, axis=(1, 2, 3), keepdims=True)  # Shape: (batch_size, 1, 1, 1)
    std = np.std(clips, axis=(1, 2, 3), keepdims=True)
    return (clips - mean) / (std + 1e-8)

# --- Normalization Selection ---
NORMALIZATION_METHODS = {
    "per_channel_dataset": normalize_per_channel_dataset,
    "global_dataset": normalize_global_dataset,
    "per_clip_channel": normalize_per_clip_channel,
    "per_clip_whole": normalize_per_clip_whole,
    None: lambda x: x,  # No normalization
}

# --- Data Loading and Batch Generation ---
def load_data_from_directory(directory, normalization=None):
    """Loads all EEG clips and labels from a given directory and applies normalization."""
    clips, labels = [], []
    h5_files = glob.glob(os.path.join(directory, "*.h5"))

    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            clips.append(np.array(f["clip"]))  # Shape: (12, 19, 100)
            labels.append(np.array(f["label"]))  # Scalar (1 or 0)

    clips = np.array(clips)  # Shape: (num_clips, 12, 19, 100)
    labels = np.array(labels)  # Shape: (num_clips,)

    # Apply normalization
    clips = NORMALIZATION_METHODS[normalization](clips)

    print(f"Loaded {clips.shape[0]} clips from {directory} with shape {clips.shape}")
    print(f"Label distribution: {np.sum(labels)} positive ({np.mean(labels) * 100:.2f}%) and {len(labels) - np.sum(labels)} negative")

    return clips, labels

def batch_generator(clips, labels, batch_size, normalization=None, shuffle_data=True):
    """Yields batches from preloaded EEG clips in memory."""
    num_clips = clips.shape[0]

    if shuffle_data:
        clips, labels = shuffle(clips, labels, random_state=42)

    for i in range(0, num_clips, batch_size):
        batch_clips = clips[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        # Apply normalization
        batch_clips = NORMALIZATION_METHODS[normalization](batch_clips)

        yield batch_clips, batch_labels

def batch_generator_from_files(directory, batch_size, normalization=None, shuffle_data=True):
    """Loads EEG clips in batches directly from .h5 files instead of preloading them all into memory."""
    h5_files = glob.glob(os.path.join(directory, "*.h5"))

    if shuffle_data:
        h5_files = shuffle(h5_files, random_state=42)

    for i in range(0, len(h5_files), batch_size):
        batch_clips, batch_labels = [], []

        for h5_file in h5_files[i:i + batch_size]:
            with h5py.File(h5_file, "r") as f:
                batch_clips.append(np.array(f["clip"]))  # Shape: (12, 19, 100)
                batch_labels.append(np.array(f["label"]))  # Scalar (1 or 0)

        batch_clips = np.array(batch_clips)  # Shape: (batch_size, 12, 19, 100)
        batch_labels = np.array(batch_labels)  # Shape: (batch_size,)

        # Apply normalization
        batch_clips = NORMALIZATION_METHODS[normalization](batch_clips)

        yield batch_clips, batch_labels

# --- Example Usage ---

batch_size = 32
train_dir = "./detection_pre/train/clipLen12_timeStepSize1"
dev_dir = "./detection_pre/dev/clipLen12_timeStepSize1"
eval_dir = "./detection_pre/eval/clipLen12_timeStepSize1"

# Test different normalizations
for norm in NORMALIZATION_METHODS.keys():
    print(f"\nUsing normalization: {norm}")

    # Load and batch train data
    train_clips, train_labels = load_data_from_directory(train_dir, normalization=norm)
    for clips_batch, labels_batch in batch_generator(train_clips, train_labels, batch_size, normalization=norm):
        print("Train batch shape:", clips_batch.shape)
        break  # Show first batch

    # Load and batch dev data
    dev_clips, dev_labels = load_data_from_directory(dev_dir, normalization=norm)
    for clips_batch, labels_batch in batch_generator(dev_clips, dev_labels, batch_size, normalization=norm):
        print("Dev batch shape:", clips_batch.shape)
        break  # Show first batch

    # Load and batch eval data
    eval_clips, eval_labels = load_data_from_directory(eval_dir, normalization=norm)
    for clips_batch, labels_batch in batch_generator(eval_clips, eval_labels, batch_size, normalization=norm):
        print("Eval batch shape:", clips_batch.shape)
        break  # Show first batch

