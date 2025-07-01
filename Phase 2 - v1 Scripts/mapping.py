import os
import pandas as pd

# Paths to the input files
input_dir = "/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/quantized_units2"
mapping_files = {
    "train": "/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/train_mapping.tsv",
    "valid": "/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/valid_mapping.tsv",
    "test": "/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/test_mapping.tsv"
}

# Function to read and update the sample IDs in the unit files
def update_sample_ids(split):
    # Load the mapping file
    mapping_path = mapping_files[split]
    mapping_df = pd.read_csv(mapping_path, sep="\t", header=None, names=["new_id", "dys_audio", "healthy_audio"])

    # Track which sample IDs have already been used
    used_sample_ids = set()

    # Create a dictionary with lists of IDs for each healthy_audio entry
    mapping_dict = {}
    for _, row in mapping_df.iterrows():
        healthy_audio = row["healthy_audio"].replace(".wav", "")
        if healthy_audio not in mapping_dict:
            mapping_dict[healthy_audio] = []
        mapping_dict[healthy_audio].append(row["new_id"])

    # Read the unit file
    unit_file_path = os.path.join(input_dir, f"{split}.txt")
    with open(unit_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Update sample IDs using the mapping, ensuring no repetition
    updated_lines = []
    for line in lines:
        sample_id, units = line.strip().split("|", maxsplit=1)
        if sample_id in mapping_dict and mapping_dict[sample_id]:
            # Get the next available sample ID
            new_sample_id = mapping_dict[sample_id].pop(0)
            if new_sample_id in used_sample_ids:
                print(f"Warning: Sample ID {new_sample_id} already used for {split}.txt")
            else:
                used_sample_ids.add(new_sample_id)
                updated_lines.append(f"{new_sample_id}|{units}\n")
        else:
            print(f"Warning: No available mapping for {sample_id} in {split}")

    # Write the updated lines back to the file
    with open(unit_file_path, "w", encoding="utf-8") as file:
        file.writelines(updated_lines)

# Process all splits
for split in ["train", "valid", "test"]:
    print(f"Updating {split}.txt...")
    update_sample_ids(split)
    print(f"{split}.txt updated successfully.")
