import json
import os
from huggingface_hub import HfApi, ModelFilter, utils
from time import sleep

model_id_to_safetensor_files: dict[str, list[str]] = {}

api = HfApi()
model_list = []
models = api.list_models(library='safetensors')
model_found = False

with open('latest-safetensor-models-text-gen.jsonl', 'r') as f:
    all_models = f.readlines()
    model_names = [json.loads(m)['model_id'] for m in all_models if m.strip() != '']
    last_model_id = model_names[-2]
    print(f'Continuing from {last_model_id}')

with open('latest-safetensor-models-text-gen.jsonl', 'a+') as f:
    for model in models:
        sleep(0.001)
        if model.modelId in model_names:
            print(f'Skipping {model.modelId}')
            continue

        model_found = True

        print(model.modelId, model.author)
        print(model.tags)

        try:
            repo_info = api.repo_info(repo_id=model.modelId)
            # print(repo_info.siblings)
            safetensors_files = [
                file.rfilename for file in repo_info.siblings if file.rfilename.endswith('safetensors')
            ]
            print(safetensors_files)
        except utils._errors.GatedRepoError as e:
            print(e)
            continue
        except utils._errors.RepositoryNotFoundError as e:
            print(e)
            continue

        f.write(
            json.dumps(
                {'model_id': model.modelId,
                    'files': sorted(safetensors_files)},
            )
        )
        f.write('\n')
