<div align="center">

<!-- TODO: Logo -->

<h1>cake 🍰</h1>

![license_shield](https://img.shields.io/github/license/simonamdev/cake?style=flat-square)
![issues_shield](https://img.shields.io/github/issues/simonamdev/cake?style=flat-square)
![pull_requests_shield](https://img.shields.io/github/issues-pr/simonamdev/cake?style=flat-square)

</div>

**cake** is an experimental, more efficient way to download and store Machine Learning models from [🤗 Hugging Face](https://huggingface.co/). Think of it as 🐋 docker, but for ML models.

Leveraging the [hugginface/safetensors](https://huggingface.co/docs/safetensors/en/index) format, it enables:

- Parallelising downloads of multiple layers at the same time.
- Robustness against network failures. `cake` caches each layer to disk, so halting half-way and retrying will not re-download already downloaded layers.
- Deduplication of layers based on their contents, even across different models. If you download `Mistral-7B-v0.1` followed by a fine-tune of it which only modified the top two layers, then `cake` will only download the top two layers.

## Roadmap

- [x] Setup linting in CI
- [x] Setup local storage based on layer hashes
- [ ] On push to `main`, build the executable and create a release
- [ ] Make CLI arguments easier to use (example: `cake download foo` instead of `cake download --model-id foo`)
- [ ] Setup config and allow overriding of storage folder, registry URL, etc
- [ ] Setup a public facing instance of the hashes registry

## Installation

Currently `cake` can only be built from source. Pre-built binaries coming soon™️.

## Usage

`cake help` to view how to use it.

`cake download --model-id <MODEL_ID>` to download a model to a folder relative to `cake` called `storage` (config coming soon™️).

Example: `cake download --model-id KoboldAI/fairseq-dense-1.3B` will download this model: https://huggingface.co/KoboldAI/fairseq-dense-1.3B from the `main` branch.

## Contributing

`cake` at this time is a personal project of mine with two main aims:

1. Introducing better tooling into ML workflows
2. Learning the `rust` programming language

Contributions targetting either of the above are appreciated and will be reviewed on a best-effort basis.

# The idea behind cake

Given a model name (example: `Mistral-7B-OpenOrca`):

1. Extract the layer hashes for the model
2. Check if all the layers are stored locally
3. Create a diff of the layers available locally and the layers required
4. For each layer required:
   1. Pull only the layers required from the remote storage [1]
   2. Compress it for local storage
5. Once all layers are available, export a new full model file

### Potential issues

As marked with [1], the "remote storage" is not fully figured out yet. Docker has the idea of a registry that could also work here. Using the `Range` HTTP header has allowed us to pull only specific layers from Huggingface so far.

Example curl: `curl --range 262175808-379616319 -L https://huggingface.co/KoboldAI/fairseq-dense-1.3B/resolve/main/model.safetensors\?download\=true -o model.safetensors`
