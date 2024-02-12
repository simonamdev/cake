import json
import os
from collections import Counter, defaultdict

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_top_layer_hashes(files):
    layer_hashes = defaultdict(Counter)

    for file_path in files:
        data = read_json(file_path)
        tensors = data.get('tensors', {})
        for tensor_name, hash_value in tensors.items():
            # Extract layer name from tensor name
            layer_name = tensor_name.split('.')[1]
            layer_hashes[layer_name][hash_value] += 1

    # Find the most common hash for each layer
    top_layer_hashes = {}
    for layer, hashes in layer_hashes.items():
        top_hash, count = hashes.most_common(1)[0]
        top_layer_hashes[layer] = (top_hash, count)

    return top_layer_hashes

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
    print("Most common layer hash and the files it is found in:")
    for layer, (hash_value, count) in top_layer_hashes.items():
        print(f"Layer: {layer}, Hash: {hash_value}, Count: {count}")
        print("Files:")
        for file_path in result_file_paths:
            data = read_json(file_path)
            tensors = data.get('tensors', {})
            for tensor_name, hash_val in tensors.items():
                if tensor_name.split('.')[1] == layer and hash_val == hash_value:
                    print(f"- {file_path}")
    print()

if __name__ == "__main__":
    main()
