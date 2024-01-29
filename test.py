from safetensors import safe_open

"""
All this does so far is prove that we can open a safetensors file and print it
"""

file_path = '/home/simon/Downloads/models/mistral-7B-v0.1/model-00001-of-00002.safetensors'

print(file_path)

with safe_open(file_path, framework='pt', device=0) as f:
    for k in f.keys():
        tensor = f.get_tensor(k)
        print(tensor)