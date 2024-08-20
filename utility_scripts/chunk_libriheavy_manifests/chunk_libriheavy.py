import os
import gzip
from tqdm import tqdm


def split_jsonl_gz_file(input_file, lines_per_chunk=10000):
    # Extract directory and base filename
    output_dir = os.path.dirname(input_file) + os.path.split(input_file)[1].split('.')[0].split('_')[-1]
    base_name = os.path.basename(input_file).rsplit('.', 2)[0]  # Remove .jsonl.gz

    with gzip.open(input_file, 'rt') as infile:
        file_count = 0
        line_count = 0
        output_file = os.path.join(output_dir, f"{base_name}_chunk_{file_count:03d}.jsonl.gz")
        outfile = gzip.open(output_file, 'wt')

        for line in tqdm(infile):
            if line_count >= lines_per_chunk:
                outfile.close()
                file_count += 1
                line_count = 0
                output_file = os.path.join(output_dir, f"{base_name}_chunk_{file_count:03d}.jsonl.gz")
                outfile = gzip.open(output_file, 'wt')
            outfile.write(line)
            line_count += 1

        outfile.close()  # Ensure the last file is closed


if __name__ == "__main__":

    input_files = ['data/libri-light/unlab/libriheavy/libriheavy_cuts_small.jsonl.gz',
                   'data/libri-light/unlab/libriheavy/libriheavy_cuts_medium.jsonl.gz',
                   'data/libri-light/unlab/libriheavy/libriheavy_cuts_large.jsonl.gz']

    for input_file in input_files:
        split_jsonl_gz_file(input_file, lines_per_chunk=100000)
