import json
import struct

from safetensors import safe_open
from pprint import pprint



file_path = '/home/simon/Downloads/models/mistral-7B-v0.1/model-00001-of-00002.safetensors'

print(file_path)


# """
# All this does so far is prove that we can open a safetensors file and print it
# """
# with safe_open(file_path, framework='pt') as f:
#     for k in f.keys():
#         tensor = f.get_tensor(k)
#         print(tensor)
#         break

"""
This reads the file according to its specifiation:
https://huggingface.co/docs/safetensors/index
"""
with open(file_path, 'rb') as f:
    header = f.read(8)
    json_header_length = struct.unpack('Q', header)[0]
    print(json_header_length)
    json_header = json.loads(f.read(json_header_length))
    pprint(json_header)
    