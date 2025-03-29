import pandas as pd
import numpy as np
from collections import defaultdict

def process_csv(file_path, output_dir):
    df = pd.read_csv(file_path)
    
    # Group by patient_id and session
    grouped = df.groupby(['patient_id', 'session'])
    
    # Prepare output file handlers
    file_handlers = {
        "train": (open(f"{output_dir}/trainSet_seq2seq_12s_sz.txt", "w"),
                   open(f"{output_dir}/trainSet_seq2seq_12s_nosz.txt", "w")),
        "dev": (open(f"{output_dir}/devSet_seq2seq_12s_sz.txt", "w"),
                 open(f"{output_dir}/devSet_seq2seq_12s_nosz.txt", "w")),
        "eval": (open(f"{output_dir}/testSet_seq2seq_12s_sz.txt", "w"),
                  open(f"{output_dir}/testSet_seq2seq_12s_nosz.txt", "w"))
    }
    
    for (patient_id, session), session_df in grouped:
        min_time = int(session_df['start_time'].min())
        max_time = int(session_df['stop_time'].max())
        
        # Store all (start_time, stop_time) intervals where binary_label = 1
        seizure_intervals = []
        for _, row in session_df.iterrows():
            if row['binary_label'] == 1:
                seizure_intervals.append((row['start_time'], row['stop_time']))
        
        # Slide a 12s window across the session
        window_size = 12
        window_step = 12
        window_index = 0
        
        for start in range(min_time, max_time - window_size + 1, window_step):
            end = start + window_size
            
            # Ensure the last window is at least 12 seconds long
            if end > max_time:
                continue
            
            # Check if window overlaps any seizure interval
            label = 0
            for s_start, s_end in seizure_intervals:
                if not (end <= s_start or start >= s_end):  # Overlap condition
                    label = 1
                    break
            
            filename = f"{patient_id}_{session}.edf_{window_index}.h5,{label}\n"
            ref_type = session_df.iloc[0]['reference_type']  # Assume same for session
            
            if ref_type in file_handlers:
                sz_file, nosz_file = file_handlers[ref_type]
                (sz_file if label == 1 else nosz_file).write(filename)
            
            window_index += 1
    
    # Close file handlers
    for files in file_handlers.values():
        for f in files:
            f.close()

# Example usage
process_csv("/local/home/nnvo/EECS 6322 - Project/DataExtract/binary_classification_labelled.csv", "/local/home/nnvo/EECS 6322 - Project/DataExtract")

