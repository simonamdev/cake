import json
from huggingface_hub import HfApi, ModelFilter, utils

model_id_to_safetensor_files: dict[str, list[str]] = {}

api = HfApi()
model_list = []
models = api.list_models(filter=ModelFilter(task='text-generation', library='safetensors'))

for model in models:
    print(model.modelId, model.author)

    try:
        repo_info = api.repo_info(repo_id=model.modelId)
        # print(repo_info.siblings)
        safetensors_files = [
            file.rfilename for file in repo_info.siblings if file.rfilename.endswith('safetensors')
        ]
        print(safetensors_files)
        model_id_to_safetensor_files[model.modelId] = sorted(safetensors_files)
    except utils._errors.GatedRepoError as e:
        print(e)

print(model_id_to_safetensor_files.keys())
print(model_id_to_safetensor_files)
print(len(model_id_to_safetensor_files))

with open('safetensor-models-text-gen.json', 'w') as f:
    f.write(
        json.dumps(
            model_id_to_safetensor_files,
            indent=2
        )
    )