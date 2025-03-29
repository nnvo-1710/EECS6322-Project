import os
import pandas as pd
import re

# Define base directory and subdirectories
BASE_DIR = "./v2.0.3/edf"
DATA_DIRS = ["train", "dev", "eval"]

# Dictionary to store extracted data
data_list = []

# First, collect all available .edf files
edf_files = set()
for directory in DATA_DIRS:
    for root, _, files in os.walk(os.path.join(BASE_DIR, directory)):
        for file in files:
            if file.endswith(".edf"):
                edf_files.add(os.path.join(root, file).replace(".edf", ""))  # Store without extension

# Now, process only corresponding _bi.csv files
for directory in DATA_DIRS:
    for root, _, files in os.walk(os.path.join(BASE_DIR, directory)):
        for file in files:
            if file.endswith(".csv"):
                file_base = os.path.join(root, file).replace(".csv", "")
                if file_base in edf_files:  # Check if matching .edf exists
                    

                    dir_split = root.split("/")
                    
                    if len(dir_split) < 10:
                        continue
                    
                    patient_id = dir_split[-3]
                    year = dir_split[-2].split("_")[-1]
                    reference_type = dir_split[-4]
                    group = dir_split[-5]
                    match = re.search(r"(s\d+_t\d+)", file_base)
                    if match:
                      session = match.group(1)  # Correct session format: sXXX_tXXX
                    else:
                      session = "UNKNOWN"
                    

                    # Read CSV file
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path, comment="#")  # Skip metadata lines
                        df["patient_id"] = patient_id
                        df["session"] = session
                        df["year"] = year
                        df["reference_type"] = reference_type
                        df["group"] = group
                        df["directory"] = directory
                        df["file_path"] = file_path

                        # Keep only relevant columns
                        df = df[["channel", "start_time", "stop_time", "label", "confidence",
                                 "patient_id", "session", "year", "reference_type", "group", "directory", "file_path"]]

                        # Append extracted data
                        data_list.append(df)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

# Concatenate all extracted DataFrames
if data_list:
    final_df = pd.concat(data_list, ignore_index=True)
    print(f"Extracted {len(final_df)} rows from {len(data_list)} files.")
else:
    final_df = pd.DataFrame()
    print("No valid files found.")

# Show summary
print(final_df.head())

# Save final DataFrame to CSV (optional)
final_df.to_csv("extracted_data_detailed.csv", index=False)

