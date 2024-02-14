from hf_transfer import download

response = download(
    url='https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/model-00001-of-00002.safetensors?download=true',
    filename='mistral7b-model-00001-of-00002.safetensors',
    max_files=128,
    chunk_size=10_485_760,
)

print(response)