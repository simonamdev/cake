from huggingface_hub import HfApi, ModelFilter

api = HfApi()
model_list = []
models = api.list_models(filter=ModelFilter(task='text-generation', library='safetensors'))
for model in models:
    print(model.modelId)
    model_list.append(model.modelId)

print(model_list)
print(len(model_list))