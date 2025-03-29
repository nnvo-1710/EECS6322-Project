import pandas as pd

def process_seizure_csv(file_path, output_dir):
    df = pd.read_csv(file_path)
    
    # Group by patient_id and session
    grouped = df.groupby(['patient_id', 'session'])
    
    # Prepare output file handlers
    file_handlers = {
        "train": open(f"{output_dir}/trainSet_seizure_files.txt", "w"),
        "dev": open(f"{output_dir}/devSet_seizure_files.txt", "w"),
        "eval": open(f"{output_dir}/testSet_seizure_files.txt", "w")
    }
    
    # Process each session
    for (patient_id, session), session_df in grouped:
        # Sort by start_time to ensure seizures are processed in order
        session_df = session_df.sort_values(by='start_time')
        
        # To store unique seizures per session based on (start_time, stop_time)
        seen_seizures = set()
        
        # Initialize seizure index
        seizure_index = 0
        
        # Iterate over the rows of the session
        for _, row in session_df.iterrows():
            # Each row represents a seizure event
            seizure_start = row['start_time']
            seizure_end = row['stop_time']
            seizure_type = row['label']
            
            # Create a unique identifier for this seizure (using start_time and stop_time)
            seizure_identifier = (seizure_start, seizure_end)
            
            # Skip if this seizure has already been processed for this session
            if seizure_identifier in seen_seizures:
                continue
            
            # Mark this seizure as seen
            seen_seizures.add(seizure_identifier)
            
            # Format the filename as per the requirement
            filename = f"{patient_id}_{session}.edf"
            
            # Get the reference type for this session
            ref_type = row['reference_type']
            
            # Write the formatted output to the appropriate file
            if ref_type in file_handlers:
                file_handlers[ref_type].write(f"{filename},{seizure_type},{seizure_index}\n")
            
            # Increment seizure index
            seizure_index += 1
    
    # Close file handlers
    for f in file_handlers.values():
        f.close()

# Example usage
process_seizure_csv("/local/home/nnvo/EECS 6322 - Project/DataExtract/Extracted_CSVs/classification_labelled_data2.csv", "/local/home/nnvo/EECS 6322 - Project/DataExtract/Filemarkers_classification")

