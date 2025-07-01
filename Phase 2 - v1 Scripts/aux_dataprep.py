import os

def convert_multitask_format_with_header(input_file, output_file, add_header=True):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Add the header if needed
        if add_header:
            outfile.write("id\ttgt_text\n")

        for line in infile:
            # Split the line by the '|' separator
            parts = line.strip().split('|')
            
            if len(parts) != 2:
                print(f"Skipping invalid line: {line.strip()}")
                continue

            # Extract sample ID and tokens
            sample_id, tokens = parts
            # Replace extra whitespace and ensure the format is tab-separated
            formatted_line = f"{sample_id.strip()}\t{tokens.strip()}\n"

            # Write to the output file
            outfile.write(formatted_line)

# Replace these paths with your actual input and output file paths
input_file_path = "/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/quantized_units_dys/valid.txt"
output_file_path = "/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/s2ut/reduced_orig_unit/valid.tsv"

# Run the conversion
convert_multitask_format_with_header(input_file_path, output_file_path)

print("Conversion completed successfully.")
