import os
import json

from tqdm import tqdm

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

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

    p_bar = tqdm(result_file_paths)
    missing_count = 0
    for file_path in p_bar:
        p_bar.set_description_str(file_path)
        data = read_json(file_path)
        if len(data) == 0 or 'compressed_size' not in data[list(data.keys())[0]].keys():
            # print(data)
            missing_count += 1
            os.remove(file_path)
            os.rmdir(os.path.dirname(file_path))
            try:
                os.rmdir(os.path.dirname(os.path.dirname(file_path)))
            except OSError:
                pass
            continue

    print(f'{missing_count} missing, {len(result_file_paths)} total')


if __name__ == "__main__":
    main()
