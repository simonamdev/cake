import os
import json

from tqdm import tqdm

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_byte_content(result_file_paths):
    saved_byte_content = 0
    total_byte_content = 0
    total_byte_content_compressed = 0
    tracked_layer_hashes = set()
    layer_count = 0
    duple_layer_count = 0

    for file_path in tqdm(result_file_paths):
        data = read_json(file_path)
        file_byte_count = 0
        file_byte_count_compressed = 0
    
        for tensor_metadata in data.values():
            layer_count += 1
            layer_hash = tensor_metadata['hash']
            offsets = tensor_metadata['data_offsets']
            # Not all files support this yet
            # byte_count = tensor_metadata['size']
            byte_count = offsets[1] - offsets[0]

            file_byte_count += byte_count
            if 'compressed_size' in tensor_metadata:
                file_byte_count_compressed += tensor_metadata['compressed_size']

            if layer_hash in tracked_layer_hashes:
                duple_layer_count += 1
                saved_byte_content += byte_count

            tracked_layer_hashes.add(layer_hash)
        
        total_byte_content += file_byte_count
        total_byte_content_compressed += file_byte_count_compressed

    num_files = len(result_file_paths)

    percent_saved = (saved_byte_content / total_byte_content) * 100 if total_byte_content != 0 else 0

    return total_byte_content, total_byte_content_compressed, num_files, saved_byte_content, percent_saved, layer_count, duple_layer_count

def bytes_to_terrabytes(bytes_value):
    return bytes_value / (1024 * 1024 * 1024 * 1024)

def main():
    result_file_paths = []
    results_directory = "../results"  # Update this with the directory containing your JSON files
    accounts = os.listdir(results_directory)
    for account in accounts:
        models = os.listdir(os.path.join('../', 'results', account))
        for model in models:
            hashes_file_path = os.path.join('../', 'results', account, model, 'hashes.json')
            if os.path.exists(hashes_file_path):
                result_file_paths.append(
                    hashes_file_path
                )

    total_byte_content, total_byte_content_compressed, num_files, bytes_saved, percent_saved, layer_count, dupe_layer_count = calculate_byte_content(result_file_paths)

    total_tb = bytes_to_terrabytes(total_byte_content)
    total_tb_compressed = bytes_to_terrabytes(total_byte_content_compressed)
    bytes_saved_tb = bytes_to_terrabytes(bytes_saved)

    print("Number of files involved:", num_files)
    print(f'Total Tensor count: {layer_count}. Dupe Tensor count: {dupe_layer_count} ({round(dupe_layer_count / layer_count * 100, 2)}%)')
    print(f"Total byte content across all files: {round(total_tb, 2)}TB", )
    print(f'Total byte content across all files when compressed: {round(total_tb_compressed, 2)}TB (NOT CORRECT DUE TO MISSING DATA')
    print(f"Amount of bytes saved if duplicate layers are deduped (absolute) {round(bytes_saved_tb, 2)}TB:")
    print("Amount of bytes saved if duplicate layers are deduped (percentage):", round(percent_saved, 2), "%")

    model_files_file_path = '../safetensor-models-text-gen.json'
    with open(model_files_file_path, 'r') as f:
        model_data = json.loads(f.read())
    
    print(f'{num_files} of {len(model_data)} files processed ({round(num_files / len(model_data) * 100, 2)}%)')

if __name__ == "__main__":
    main()
