#!/usr/bin/env python3

import json
import argparse

def process_units(units):
    """Remove consecutive duplicate units while maintaining order."""
    unit_list = units.split()
    processed_units = [unit_list[i] for i in range(len(unit_list)) if i == 0 or unit_list[i] != unit_list[i - 1]]
    return " ".join(processed_units)

def process_hubert_file(input_file, output_file):
    """Read the input file, process 'hubert' values, and save the modified content to output file."""
    processed_data = []

    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            json_string = line.strip().replace("'", "\"")  # Convert single quotes to double quotes
            data = json.loads(json_string)  # Load each line as JSON
            
            data["hubert"] = process_units(data["hubert"])  # Process 'hubert' field
            processed_data.append(data)

    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry) + "\n")  # Save in JSON format

    print(f"Processed file saved at: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process hubert key in train.txt to remove consecutive duplicate units.")
    parser.add_argument("input_file", type=str, help="Path to input train.txt file")
    parser.add_argument("output_file", type=str, help="Path to save processed output file")

    args = parser.parse_args()
    process_hubert_file(args.input_file, args.output_file)
