import json
import os
from collections import Counter, defaultdict

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_layer_sizes(files):
    layer_sizes = {}

    for file_path in files:
        data = read_json(file_path)
        tensors = data.get('tensors', {})
        for tensor_name, tensor_data in tensors.items():
            # Extract layer name from tensor name
            hash = tensor_data['hash']
            byte_count = tensor_data['byte_count']
            layer_sizes[hash] = byte_count

    return layer_sizes

def get_top_layer_hashes(files):
    layer_hashes = {}

    for file_path in files:
        data = read_json(file_path)
        tensors = data.get('tensors', {})
        for tensor_name, tensor_data in tensors.items():
            # Extract layer name from tensor name
            hash = tensor_data['hash']
            byte_count = tensor_data['byte_count']
            if byte_count == 0:
                continue
            if hash not in layer_hashes:
                layer_hashes[hash] = 1
            else:
                layer_hashes[hash] += 1

    return layer_hashes

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
    top_layer_hashes = get_top_layer_hashes(result_file_paths)
    layer_sizes = get_layer_sizes(result_file_paths)
    print("Most common layer hash and the files it is found in:")
    print_top_n(top_layer_hashes, 10, layer_sizes)

def print_top_n(dictionary, n, layer_sizes):
    sorted_items = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    total_savings_in_mb = 0
    total_size = 0
    for hash, occurrences in sorted_items:
        total_size += layer_sizes[hash] * occurrences / 1024 / 1024
        if occurrences > 1:
            total_savings_in_mb += layer_sizes[hash] * (occurrences-1) / 1024 / 1024
    top_n = sorted_items[:n]
    for hash, occurrences in top_n:
        savings_in_mb = layer_sizes[hash] * occurrences / 1024 / 1024
        print(f"{hash}: {occurrences} occurrences (total {savings_in_mb}MB)")
    perc_saved = total_savings_in_mb / total_size * 100
    print(f'Total Size: {total_size}MB Total savings in MB: {total_savings_in_mb}. Only {perc_saved}% saved across {len(dictionary)} layers')

if __name__ == "__main__":
    main()
