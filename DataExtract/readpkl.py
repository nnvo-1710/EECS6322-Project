import pickle
import argparse
import os

def read_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    print(f"Data Type: {type(data)}")
    
    if isinstance(data, dict):
        print("Keys:", list(data.keys())[:5])
        print("Sample Values:", {k: data[k] for k in list(data.keys())[:5]})
    elif isinstance(data, list):
        print("Sample Elements:", data[:5])
    else:
        print("Data Sample:", data)

if __name__ == "__main__":
    # Set up the argument parser to take a file path
    parser = argparse.ArgumentParser(description="Process a pickle file.")
    parser.add_argument('file_path', type=str, help="Path to the pickle file")
    args = parser.parse_args()

    # Check if the file exists
    if not os.path.isfile(args.file_path):
        print(f"Error: The file '{args.file_path}' does not exist.")
        exit(1)

    # Call the function to read the pickle file
    read_pkl(args.file_path)

