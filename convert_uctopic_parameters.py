#%%
import torch
from tqdm import tqdm
from transformers import LukeTokenizer, LukeModel, LukeConfig
from uctopic.models import UCTopic
#%%
model = LukeModel.from_pretrained("studio-ousia/luke-base")
model_state_dict = model.state_dict()
print(model_state_dict.keys())

#%%
device = torch.device('cpu')
uctopic = torch.load('result/uctopic_base_3/pytorch_model.bin', map_location=device)
print(uctopic.keys())
# %%
luke_config = LukeConfig.from_pretrained("studio-ousia/luke-base")
uctopic = UCTopic(luke_config)
print(uctopic.state_dict().keys())
# %%
state_dict = torch.load('result/uctopic_base_3/pytorch_model.bin', map_location=torch.device("cpu"))
new_state_dict = {}
for key, param in state_dict.items():
    if 'luke' in key or 'mlp' in key:
        new_state_dict[key] = param

torch.save(new_state_dict, 'result/pytorch_model.bin')

# %%
uctopic.load_state_dict(torch.load('result/pytorch_model.bin', map_location=device))
# %%
