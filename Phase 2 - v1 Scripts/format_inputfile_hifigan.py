import os
import soundfile as sf

# Define file paths
quantized_units_file = "/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/quantized_units2_copy/test.txt"  # Replace with actual path
audio_base_dir = "/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/healthy"  # Replace with actual path
output_file = "/nlsasfs/home/nltm-st/sanb/fairseq/examples/textless_nlp/dsr/data/output/quant_units/test.txt"  # Replace with desired output path

# Helper function to get audio duration
def get_audio_duration(audio_path):
    try:
        with sf.SoundFile(audio_path) as audio_file:
            return audio_file.frames / audio_file.samplerate
    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")
        return 0.0

# Read quantized units file and generate output
with open(quantized_units_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Parse each line in the quantized units file
        audio_id, units = line.strip().split("|")
        audio_path = os.path.join(audio_base_dir, f"{audio_id}.wav")

        # Check if the audio file exists
        if not os.path.exists(audio_path):
            print(f"Audio file {audio_path} not found. Skipping.")
            continue

        # Get duration of the audio file
        duration = get_audio_duration(audio_path)

        # Format the output dictionary
        output_dict = {
            "audio": audio_path,
            "hubert": units.strip(),
            "duration": round(duration, 2),
        }

        # Write the dictionary to the output file
        outfile.write(str(output_dict) + "\n")

print(f"Output file generated at: {output_file}")
