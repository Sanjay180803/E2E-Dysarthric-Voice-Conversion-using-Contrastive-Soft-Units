import sys
from collections import defaultdict

def create_dict_with_frequencies(input_files, output_file):
    token_counts = defaultdict(int)

    # Read each file and count the frequency of each token
    for file_path in input_files:
        with open(file_path, 'r') as file:
            for line in file:
                tokens = line.strip().split()
                for token in tokens:
                    token_counts[token] += 1

    # Sort the tokens by numerical value and write to the output file
    sorted_tokens = sorted(token_counts.items(), key=lambda x: int(x[0]))
    with open(output_file, 'w') as file:
        for token, count in sorted_tokens:
            file.write(f"{token} {count}\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_dict.py <output_file> <input_file1> <input_file2> ...")
        sys.exit(1)

    output_file = sys.argv[1]
    input_files = sys.argv[2:]
    create_dict_with_frequencies(input_files, output_file)
