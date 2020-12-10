import sys
sys.path.insert(1, '/home/louis/projU2IS/controller/vae')
sys.path.insert(1, '/home/louis/projU2IS/controller/trained_vaes')
sys.path.insert(1, '/home/louis/projU2IS/controller/data')
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from test import d
from dataset_structures import DataRecord
from vaes import VAELow, val_vae, train_vae
import torch
import pickle

from torch.utils.data import DataLoader
agent_name = 'robot'
sensor_name = 'IR_1'

p_file = "data/recordings.p"

eval_data = DataRecord(p_file, train=False, train_rate=0.8, step_sample=1, step_dot=1)
eval_loaded = DataLoader(eval_data, batch_size=64, shuffle=True) #Test pour vérifier que le VAE fonctionne sur le deep sensor. loss: 5 -> OK

#PATH = "vae/VAELow_25epochs_60000steps_NEW_ENVIRONMENT__batch64_Adam_lr7E-3.pt"
PATH="trained_vaes/VAELow.pt"
encoder = VAELow()
#val_vae(encoder, eval_loaded, agent_name, sensor_name)
state_dict_vae = torch.load(PATH)['state_dict']
encoder.load_state_dict(state_dict_vae)
val_vae(encoder, eval_loaded, agent_name, sensor_name)