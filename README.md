<div align="center">

<!-- TODO: Logo -->

<h1>cake 🍰</h1>

![license_shield](https://img.shields.io/github/license/simonamdev/cake?style=flat-square)
![issues_shield](https://img.shields.io/github/issues/simonamdev/cake?style=flat-square)
![pull_requests_shield](https://img.shields.io/github/issues-pr/simonamdev/cake?style=flat-square)


</div>

**cake** is an experimental, more efficient way to download and store Machine Learning models from [🤗 Hugging Face](https://huggingface.co/). Think of it as 🐋 docker, but for ML models.

Leveraging the [hugginface/safetensors](https://huggingface.co/docs/safetensors/en/index) format, it enables:
* Usage of more of a machine's network bandwidth by parallelising downloads of multiple tensors at the same time.
* Robustness against network failures. `cake` caches each layer to disk, so stopping half-way and retrying will not re-download already downloaded layers.
* (❕COMING SOON™️❕) Deduplication of layers based on their contents. If you download `Mistral-7B-v0.1` followed by a fine-tune of it which only modified the top two layers, then `cake` will only download the top two layers.

## Roadmap

[] Version 0.1:
* Setup linting in CI.
* Build the executable on push to `main` branch.
* Open source!

[] Version 0.2
* Setup a TUI interface to enable listing / searching of available models

[] Version 0.3:
* Setup a server component that will serve hashes for a given model
* Change local storage to store tensors based on their hashes
* Add deduplication of downloading hashed tensors. If we already have the tensor, do not download it.

[] Version 0.4:
* Download and hash at least 25% of text-gen models from huggingface hub.
* Write a blog post about the results

## Installation

Currently `cake` can only be built from source. Pre-built binaries coming soon™️.

## Contributing

`cake` at this time is a personal project of mine with two main aims:
1. Learning the `rust` programming language
2. Introducing better tooling into ML workflows

As a result, I plan on loosely following the above roadmap at my own pace. If you wish to contribute, feel free to open a PR which will either:
1. Attempt to prototype the next part of the roadmap.
2. Shows me how terrible my rust skills are by doing what I did but better.

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