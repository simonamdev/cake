import json
from huggingface_hub import HfApi, ModelFilter, utils

model_id_to_safetensor_files: dict[str, list[str]] = {}

api = HfApi()
model_list = []
models = api.list_models(library='safetensors')
model_found = False

with open('latest-safetensor-models-text-gen.jsonl', 'a+') as f:
    for model in models:

        print(model.modelId, model.author)
        print(model.tags)

        if not model_found and not model.modelId == 'Coelhomatias/deit-cvc-drop-aug':
            continue

        model_found = True

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
