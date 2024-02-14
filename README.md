# cake

An experimental, more efficient way to download and store Machine Learning models.

# Proposal

The idea behind cake is that distribution of ML models right now is a very slow process: it involves downloading the entire model from a website such as Huggingface from start to end. Given the rate of creation of models and the culture of trying out different models or finetunes often, this results in large amounts of duplicated weights, both on disk and in transfer.

Cake takes a similar approach to ML models that docker does to container images: each layer is treated independently and if you already have a layer it will reuse it rather than re-retrieve it.

Using [huggingface/safetensors](https://huggingface.co/docs/safetensors/index) it will only download the specific missing tensors required to build the model you have requested.

# Process

Given a model name (example: `Mistral-7B-OpenOrca`):
1. Extract the layer hashes for the model
2. Check if all the layers are stored locally
3. Create a diff of the layers available locally and the layers required
4. For each layer required:
    1. Pull the layer required from the remote storage [1]
    2. Extract and store it locally
5. Once all layers are available, export a new full model file

# Potential issues

1. As marked with [1], the "remote storage" is not figured out yet. Docker has the idea of a registry that could also work here, however if we are able to read and pull only **parts** of `safetensors` files from an HTTP endpoint (if it allows us to request the file at an offset for a specific length) then it is possible we can get away with skipping the complexity of setting up a registry too.

It looks like snapshot_download from huggingface_hub only allows retrieval of the entire file.

We may be able to however use the Range header, but it needs further testing. Example curl: `curl --range 262175808-379616319 -L https://huggingface.co/KoboldAI/fairseq-dense-1.3B/resolve/main/model.safetensors\?download\=true -o model.safetensors`

2. It may be possible, as a future enhancement, to do a similarity comparison of layers instead of a full hash. Imagine the case where a single bit is flipped through fine-tuning - the hash will be completely different however the final result of using that layer may be acceptable performance wise.