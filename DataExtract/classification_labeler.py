import pandas as pd

# Load the original CSV file (replace with the correct filename)
input_csv = "extracted_data_detailed.csv"
output_csv = "classification_labelled_data2.csv"

# Define label mapping
label_map = {
    "fnsz": 0, "spsz": 0, "cpsz": 0,  # Label 0
    "gnsz": 1,                         # Label 1
    "absz": 2,                         # Label 2
    "tnsz": 3, "cnsz": 3               # Label 2
}

# Read the CSV file
df = pd.read_csv(input_csv)

# Filter and replace labels based on the mapping
df_filtered = df[df["label"].isin(label_map.keys())].copy()  # Explicit copy to prevent warnings

# Apply label mapping using .loc to avoid warnings
df_filtered.loc[:, "new_label"] = df_filtered["label"].map(label_map)

# Drop old label column and rename new_label → label
df_filtered.drop(columns=["label"], inplace=True)
df_filtered.rename(columns={"new_label": "label"}, inplace=True)

# Save the cleaned data to a new CSV file
df_filtered.to_csv(output_csv, index=False)

print(f"Filtered data saved to {output_csv} with {len(df_filtered)} entries.")

