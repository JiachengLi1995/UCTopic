#%%
import torch

# %%
state_dict = torch.load('result/uctopic_base_pretraining/pytorch_model.bin', map_location=torch.device("cpu"))
new_state_dict = {}
for key, param in state_dict.items():
    if 'luke' in key or 'mlp' in key:
        new_state_dict[key] = param

torch.save(new_state_dict, 'result/uctopic_base/pytorch_model.bin')