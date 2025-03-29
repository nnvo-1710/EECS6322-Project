import pandas as pd

# Load the original CSV file
input_csv = "extracted_data_detailed.csv"  # Ensure this file exists
output_csv = "binary_classification_labelled.csv"

# Define seizure-related labels (label = 1), others will be 0
seizure_labels = {"SEIZ", "FNSZ", "GNSZ", "SPSZ", "CPSZ", "ABSZ", 
                  "TNSZ", "CNSZ", "TCSZ", "ATSZ", "MYSZ", "NESZ"}

# Read the CSV file
df = pd.read_csv(input_csv)

# Debugging: Check available columns
print("Available columns in CSV:", df.columns)

# Ensure the label column exists
if "label" not in df.columns:
    raise KeyError("Column 'label' not found in CSV. Check column names.")

# Normalize labels: Strip spaces and make uppercase for consistency
df["normalized_label"] = df["label"].astype(str).str.strip().str.upper()

# Create a new filtered DataFrame with only necessary columns
df_filtered = df.copy()

# Assign new labels (1 for seizures, 0 for others)
df_filtered["binary_label"] = df_filtered["normalized_label"].apply(lambda x: 1 if x in seizure_labels else 0)

# Drop the temporary normalized label column
df_filtered.drop(columns=["normalized_label"], inplace=True)
df_filtered.drop(columns=["label"], inplace=True)
# Save to new CSV
df_filtered.to_csv(output_csv, index=False)

print(f"Processed data saved to {output_csv} with {len(df_filtered)} entries.")

