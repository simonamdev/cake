import os

cache_files = os.listdir('./cache')

header_length_file = './cache/header.length'
header_file = './cache/header.json'

with open(header_length_file, 'r') as f:
    header_length = f.read()

with open(header_file, 'r') as f:
    header = f.read()

print(header)
# THese don't match due to whitespace inconsistencies.
# I should really just use safetensors to de/serialise these bytes instead
print(len(header))
print(int(header_length))


