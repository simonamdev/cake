from hashlib import md5
import os
import json
from tqdm import tqdm
from collections import Counter

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.loads(f.read())
    return data

def count_layer_matches(file_paths):
    all_models = []
    for file_path in file_paths:
        all_models.append(read_json(file_path))
    results = {}
    for (file_path_a, model_a) in tqdm(zip(file_paths, all_models), total=len(all_models)):
        for (file_path_b, model_b) in zip(file_paths, all_models):
            if file_path_a == file_path_b:
                continue
            model_a_hashes = [
                d['hash'] for d in model_a.values()
            ]
            model_b_hashes = [
                d['hash'] for d in model_b.values()
            ]
            match_count = len(set(model_a_hashes) & set(model_b_hashes))
            if (file_path_a, file_path_b, len(model_a_hashes), match_count) in results:
                continue
            results[(file_path_a, file_path_b, len(model_a_hashes), match_count)] = match_count/len(model_a_hashes)*100.0
    return results

def calculate_layer_size_mb(file_path):
    model = read_json(file_path)
    total_layer_size_bytes = sum([d['data_offsets'][1] - d['data_offsets'][0] for d in model.values()])
    return total_layer_size_bytes / 1024.0 / 1024.0

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
    
    results = count_layer_matches(result_file_paths)
    c = Counter(results)
    for k, v in c.most_common(20):
        similar_perc = round(v, 2)
        layer_size_mb = calculate_layer_size_mb(k[0])
        print(f'{k[0]} and {k[1]}: [{k[3]}/{k[2]} similar] {similar_perc}% ({round(layer_size_mb, 2)}MB -> {round(layer_size_mb*((100-similar_perc)/100.0), 2)}MB approx)')

    threshold = 70 # percent similar
    count_over_threshold = 0
    for v in c.values():
        similar_perc = round(v, 2)
        if similar_perc >= threshold:
            count_over_threshold += 1
    perc_models_over_threshold = round(count_over_threshold / len(result_file_paths) * 100.0, 2)
    print(f'{count_over_threshold} models out of {len(result_file_paths)} ({perc_models_over_threshold}%) are over {threshold}% similar')

if __name__ == "__main__":
    main()
