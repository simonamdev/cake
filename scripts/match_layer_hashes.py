import json
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def compare_tensors(files):
    tensors_list = []
    print('Retrieving Tensors...')
    for file_path in files:
        data = read_json(file_path)
        tensors_list.append(data['tensors'])

    num_files = len(files)
    similarity_matrix = np.zeros((num_files, num_files))

    print('Calculating similarity matrices...')
    for i in range(num_files):
        for j in range(i + 1, num_files):
            tensor1 = tensors_list[i]
            tensor2 = tensors_list[j]

            matching_hashes = sum(1 for hash_val in tensor1.values() if hash_val in tensor2.values())
            similarity = matching_hashes / max(len(tensor1), len(tensor2))
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity

    return similarity_matrix

def plot_heatmap(similarity_matrix, files):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(similarity_matrix, annot=True, xticklabels=files, yticklabels=files, cmap="YlGnBu")
    plt.title("Similarity Heatmap of Tensors Across Files")
    plt.xlabel("File")
    plt.ylabel("File")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

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
    similarity_matrix = compare_tensors(result_file_paths)
    plot_heatmap(similarity_matrix, result_file_paths)

if __name__ == "__main__":
    main()
