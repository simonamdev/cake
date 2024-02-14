from io import BufferedReader
import json
import struct

from safetensors import safe_open
from pprint import pprint
from tqdm import tqdm
import hashlib


file_path_one = '/home/simon/Downloads/models/mistral-7B-v0.1/model-00001-of-00002.safetensors'
file_path_two = '/home/simon/Downloads/models/mistral-7B-v0.1/model-00002-of-00002.safetensors'

print(file_path_one)
print(file_path_two)

def read_bytes(f: BufferedReader, start: int, length: int) -> bytes:
    f.seek(start)
    content = f.read(length // 8)
    return content
    
def hash_safetensors_file(file_path: str) -> dict:
    output = {
        'file_path': file_path,
        'tensors': {}
    }
    with open(file_path, 'rb') as f:
        # First eight bytes is the u64 number defining the amount
        # of bytes dedicated to the JSON formatted header of the file
        header = f.read(8)
        header_length_bytes = struct.unpack('Q', header)[0]
        json_header_bytes = f.read(header_length_bytes)
        json_header = json.loads(
            json_header_bytes
        )
        print(json_header)
        print(header_length_bytes)
        tensor_names = [
            t for t in list(json_header.keys()) if not t == '__metadata__' # skip the metadata
        ]
        # print(tensor_names)
        p_bar = tqdm(tensor_names)
        for tensor_name in p_bar:
            p_bar.set_description_str(tensor_name)
            # print(json_header[tensor_name])
            offset_start, offset_end = json_header[tensor_name]['data_offsets']
            # print(tensor_name, offset_start, offset_end)
            tensor_bytes_length = offset_end - offset_start
            tensor_bytes = read_bytes(f, offset_start, tensor_bytes_length)
            tensor_hash = hashlib.sha256(tensor_bytes).hexdigest()
            # print(tensor_name, tensor_hash, tensor_hash == hash(tensor_bytes))
            output['tensors'][tensor_name] = tensor_hash
    return output


def merge_results(st_one:dict, st_two:dict) -> dict:
    tensors = {}
    tensors.update(st_one['tensors'])
    tensors.update(st_two['tensors'])
    return {
        'file_paths': [
            st_one['file_path'],
            st_two['file_path']
        ],
        'tensors': tensors
    }

one = hash_safetensors_file(file_path_one)
print(one)
two = hash_safetensors_file(file_path_two)
print(two)
merged = merge_results(one, two)
print(merged)

with open('output.json', 'w') as f:
    f.write(
        json.dumps(
            merged,
            indent=2
        )
    )