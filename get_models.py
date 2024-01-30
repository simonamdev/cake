import os
from huggingface_hub import HfApi, ModelFilter, snapshot_download, utils

api = HfApi()

download_folder = '/media/simon/models'

models = api.list_models(filter=ModelFilter(tags=['pipeline_tag=text-generation', 'library=safetensors']))
for model in models:
    print(model.modelId)
    try:
        snapshot_download(
            repo_id=model.modelId,
            repo_type='model',
            allow_patterns=['*.json', '*.safetensors'],
            local_dir_use_symlinks=True,
            local_dir=os.path.join(download_folder, 'results', model.modelId),
            cache_dir=os.path.join(download_folder, 'cache')
        )
    except utils._errors.GatedRepoError as e:
        print(e)
        continue
    except utils._errors.HfHubHTTPError as e:
        print(e)
        continue