import os 
import sys
import argparse
import h5py
import glob
import re
from tqdm import tqdm
from dataloader_detection import computeSliceMatrix

FILE_MARKER_DIR = "/local/home/nnvo/EECS 6322 - Project/DataExtract/file_markers_detection"

def load_file_markers(filename):
    """Loads file marker lists from text files."""
    with open(os.path.join(FILE_MARKER_DIR, filename), "r") as f:
        return [line.strip().split(',') for line in f.readlines()]

def main(resample_dir, raw_data_dir, output_dir, clip_len, time_step_size, is_fft=True):
    """Main function for preprocessing EEG data."""
    
    # Load file markers
    train_tuples = load_file_markers(f"trainSet_seq2seq_{clip_len}s_sz.txt") + \
                   load_file_markers(f"trainSet_seq2seq_{clip_len}s_nosz.txt")
    dev_tuples = load_file_markers(f"devSet_seq2seq_{clip_len}s_sz.txt") + \
                 load_file_markers(f"devSet_seq2seq_{clip_len}s_nosz.txt")
    test_tuples = load_file_markers(f"testSet_seq2seq_{clip_len}s_sz.txt") + \
                  load_file_markers(f"testSet_seq2seq_{clip_len}s_nosz.txt")
    
    all_tuples = test_tuples 
    #+ dev_tuples + test_tuples

    # Collect all raw EDF file paths
    edf_files = {}
    for path, _, files in os.walk(raw_data_dir):
        for name in files:
            if name.endswith(".edf"):
                edf_files[name] = os.path.join(path, name)

    # Ensure output directory exists
    output_dir = os.path.join(output_dir, f'clipLen{clip_len}_timeStepSize{time_step_size}')
    os.makedirs(output_dir, exist_ok=True)

    for idx in tqdm(range(len(all_tuples))):
        h5_fn, label_str = all_tuples[idx]
        label = int(label_str)  # Convert label from string to integer
        
        # Extract the base EDF filename
        expected_edf = h5_fn.split('.edf_')[0] + ".edf"
    
        # Verify EDF file existence
        if expected_edf not in edf_files:
            raise ValueError(f"EDF file {expected_edf} not found in raw_data_dir.")

        edf_fn_full = edf_files[expected_edf]

        # Extract the correct `.h5` filename (without clip index)
        h5_fn_base = h5_fn.split('.edf_')[0] + ".h5"  # Removes `_<clip_idx>.h5`

        # Resolve the actual path of `.h5` file inside `resample_dir` (searching all subdirs)
        h5_path_candidates = glob.glob(os.path.join(resample_dir, "**", h5_fn_base), recursive=True)

        if not h5_path_candidates:
            print(f"Warning: H5 file {h5_fn_base} not found in any subdirectory of {resample_dir}. Skipping.")
            continue
        elif len(h5_path_candidates) > 1:
            print(f"Warning: Multiple matches found for {h5_fn_base}, using first match.")

        h5_fn_full = h5_path_candidates[0]

        # Extract clip index using regex
        match = re.search(r'_(\d+)\.h5$', h5_fn)
        if match:
            clip_idx = int(match.group(1))
        else:
            raise ValueError(f"Could not extract clip index from filename: {h5_fn}")

        # Compute EEG clip with label
        eeg_clip, _ = computeSliceMatrix(
            h5_fn=h5_fn_full,
            edf_fn=edf_fn_full,
            clip_idx=clip_idx,
            label=label,
            time_step_size=1,
            clip_len=clip_len,
            is_fft=is_fft,
        )

        # Save output
        output_h5_path = os.path.join(output_dir, f"{expected_edf}_{clip_idx}.h5")
        with h5py.File(output_h5_path, 'w') as hf:
            hf.create_dataset('clip', data=eeg_clip)
            hf.create_dataset('label', data=label)  # Save label in H5 file

    print("Preprocessing DONE.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resampled_dir", type=str, required=True, help="Directory to resampled signals.")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Directory to raw edf files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--clip_len", type=int, default=60, help="EEG clip length in seconds.")
    parser.add_argument("--time_step_size", type=int, default=1, help="Time step size in seconds.")
    parser.add_argument("--is_fft", action="store_true", help="Whether to perform FFT.")

    args = parser.parse_args()
    main(
        args.resampled_dir,
        args.raw_data_dir,
        args.output_dir,
        args.clip_len,
        args.time_step_size,
        args.is_fft
    )

