import os
from huggingface_hub import HfApi, ModelFilter, snapshot_download, utils

api = HfApi()

download_folder = '/media/simon/models3'
model_list_file_path = './models_downloaded.txt'

if not os.path.isfile(model_list_file_path):
    open(model_list_file_path, 'w').close()

already_downloaded_model_ids = []
with open(model_list_file_path, 'r') as f:
    for line in f:
        already_downloaded_model_ids.append(
            line.strip().replace('\n', '')
        )

models = api.list_models(filter=ModelFilter(task='text-generation', library='safetensors'))
for model in models:
    print(model.modelId)
    if model.modelId in already_downloaded_model_ids:
        continue
    try:
        force_download =False
        resume_download = True
        # force_download = model.modelId == 'bigscience/bloomz'
        # resume_download = model.modelId != 'bigscience/bloomz'
        snapshot_download(
            repo_id=model.modelId,
            repo_type='model',
            allow_patterns=['*.safetensors'],
            local_dir_use_symlinks=True,
            local_dir=os.path.join(download_folder, 'results', model.modelId),
            cache_dir=os.path.join(download_folder, 'cache'),
            force_download=force_download,
            resume_download=resume_download,
            max_workers=2
        )
        with open(model_list_file_path, 'a') as f:
            f.write(f'{model.modelId}\n')
    except utils._errors.GatedRepoError as e:
        print(e)
        continue
    except utils._errors.HfHubHTTPError as e:
        print(e)
        continue
    except OSError as e:
        print(e)
        continue