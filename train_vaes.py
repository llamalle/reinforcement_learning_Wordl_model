import sys
sys.path.insert(1, '/home/louis/projU2IS/controller/vae')
sys.path.insert(1, '/home/louis/projU2IS/controller/data')
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from dataset_structures import DataRecord
from vaes import VAELow, val_vae, train_vae, losses_training
import torch
import pickle
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



SAVE_VAE = True
agent_name = 'robot'
sensor_name = 'IR_1'

p_file = "data/new_recordings.p"
train_data = DataRecord(p_file, train=True, train_rate=0.8, step_sample=1, step_dot=1)
train_loaded = DataLoader(train_data, batch_size=32, shuffle=True)

eval_data = DataRecord(p_file, train=False, train_rate=0.8, step_sample=1, step_dot=1)
eval_loaded = DataLoader(eval_data, batch_size=32, shuffle=True)
encoder = VAELow()

optimizer = optim.Adam(encoder.parameters(), lr=0.007)
scheduler = StepLR(optimizer, step_size=3, gamma=0.7)

nb_epochs = 50
for epoch in range(1, nb_epochs + 1):
    train_vae(encoder, optimizer, epoch, train_loaded, agent_name, sensor_name)
    scheduler.step()

val_vae(encoder, eval_loaded, agent_name, sensor_name)

#plt.plot(losses_training)
if SAVE_VAE:
    name = "VAELow_new"
    torch.save({'state_dict': encoder.state_dict()}, "trained_vaes/"+name+".pt")
    #plt.savefig("trained_vaes/training_loss/"+name)
#plt.show()
