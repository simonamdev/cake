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
    

"""
Taking an example layer:
 'model.layers.7.mlp.down_proj.weight': {'data_offsets': [8634294272,
                                                          8751734784],
                                         'dtype': 'BF16',
                                         'shape': [4096, 14336]},
"""

with open(file_path, 'rb') as f:
    header = f.read(8)
    json_header_length = struct.unpack('Q', header)[0]
    print(json_header_length)
    json_header = json.loads(f.read(json_header_length))
    test_layer = json_header['model.layers.7.mlp.down_proj.weight']
    print(test_layer)
    offset_start, offset_end = test_layer['data_offsets']
    print(offset_start, offset_end)
    offset_length = offset_end - offset_start
    print(offset_length)
    f.seek(offset_start)
    tensor = f.read(offset_length)
    print('First 50:')
    print(tensor[:50])
    print(f'Tensor is {len(tensor) / 1024.0 / 1024.0} MB large')
    
