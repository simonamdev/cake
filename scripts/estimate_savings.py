import os
import json

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_byte_content(result_file_paths):
    saved_byte_content = 0
    total_byte_content = 0
    tracked_layer_hashes = set()

    for file_path in result_file_paths:
        data = read_json(file_path)
        tensors = data.get('tensors', {})
        for tensor_info in tensors.values():
            layer_hash = tensor_info['hash']
            byte_count = tensor_info['byte_count']

            total_byte_content += byte_count

            if layer_hash in tracked_layer_hashes:
                saved_byte_content += byte_count

            tracked_layer_hashes.add(layer_hash)
                

    num_files = len(result_file_paths)

    percent_saved = (saved_byte_content / total_byte_content) * 100 if total_byte_content != 0 else 0

    return total_byte_content, num_files, saved_byte_content, percent_saved

def bytes_to_megabytes(bytes_value):
    return bytes_value / (1024 * 1024)

def main():
    result_file_paths = []
    results_directory = "../results"  # Update this with the directory containing your JSON files
    accounts = os.listdir(results_directory)
    for account in accounts:
        models = os.listdir(os.path.join('../', 'results', account))
        for model in models:
            result_file_paths.append(
                os.path.join('../', 'results', account, model, 'hashes.json')
            )

    total_byte_content, num_files, bytes_saved, percent_saved = calculate_byte_content(result_file_paths)

    total_mb = bytes_to_megabytes(total_byte_content)
    bytes_saved_mb = bytes_to_megabytes(bytes_saved)

    print("Total byte content across all files (MB):", total_mb)
    print("Number of files involved:", num_files)
    print("Amount of bytes saved if duplicate layers are deduped (absolute) (MB):", bytes_saved_mb)
    print("Amount of bytes saved if duplicate layers are deduped (percentage):", percent_saved, "%")

if __name__ == "__main__":
    main()
