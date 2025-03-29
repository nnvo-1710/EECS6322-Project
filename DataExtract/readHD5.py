import h5py
import argparse
import os

# Set up the argument parser to take a directory path
parser = argparse.ArgumentParser(description="Process EEG data from an H5 file.")
parser.add_argument('directory', type=str, help="Directory containing the H5 files")
args = parser.parse_args()

# Check if the directory exists
if not os.path.isdir(args.directory):
    print(f"Error: The directory '{args.directory}' does not exist.")
    exit(1)

# Construct the path to the H5 file (example file)
h5_file_path = os.path.join(args.directory, "aaaaasfw_s007_t000.h5")

# Open the H5 file in read mode
with h5py.File(h5_file_path, "r") as h5f:
    print("Keys in the file:", list(h5f.keys()))  # Lists the datasets/groups

    # Access a dataset
    dataset = h5f["resampled_signal"][10:18][0:1200]
    print("Dataset shape:", dataset.shape)
    print("First few values:", dataset[:][100:200])

