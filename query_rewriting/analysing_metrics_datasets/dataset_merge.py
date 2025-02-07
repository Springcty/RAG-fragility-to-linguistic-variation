import os
import json
import glob

# Set the directory containing JSON files
directory = "/home/neelbhan/QueryLinguistic/dataset/dev/"  # Change this if needed

# Get all JSON files in the directory
json_files = glob.glob(os.path.join(directory, "*.json"))

# List to store merged JSON data
merged_data = []

# Read each JSON file and append its contents to the merged list
for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):  # Ensure it's a list before merging
                merged_data.extend(data)  # Flatten the list
            else:
                print(f"Skipping {file}: Expected a list but got {type(data)}")
        except json.JSONDecodeError as e:
            print(f"Error reading {file}: {e}")

# Save merged JSON data
output_file = "merged.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=4)

print(f"Merged {len(json_files)} JSON files into {output_file}")