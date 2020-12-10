#from worldmodels.data.dataset_structures import DataRecord, IRRecordSequences
#import worldmodels.vaes.vaes as vaes
#import worldmodels.rnns.gru as gru

import sys
sys.path.insert(1,'/home/louis/projU2IS/controller/rnn')
sys.path.insert(1,'/home/louis/projU2IS/controller/data')
sys.path.insert(1, '/home/louis/projU2IS/controller/vae')

from gru import GRUNet, val_gru, train_gru, losses_training
from dataset_structures import DataRecord, IRRecordSequences
from vaes import VAELow, val_vae
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

SAVE_GRU = True
agent_name = 'robot'
sensor_name = 'IR_1'
min_observation_data_VAE = 0.26
max_observation_data_VAE = 286

p_file = "data/new_recordings.p"

#########################TRAIN DATA#############################################
train_IRdata = IRRecordSequences(p_file, min_observation_data_VAE, max_observation_data_VAE, train=True,
                                 seq_in=16, seq_out=1, agent_name=agent_name)
train_IRdata_loaded = DataLoader(train_IRdata, batch_size=128, shuffle=True, drop_last=True)

############################EVAL DATA###############################################"
eval_IRdata = IRRecordSequences(p_file, min_observation_data_VAE, max_observation_data_VAE, train=False,
                                seq_in=16, seq_out=1, agent_name=agent_name)
eval_IRdata_loaded = DataLoader(eval_IRdata, batch_size=32, shuffle=True, drop_last=True)

########################LOAD VAE####################################
encoder = VAELow()
PATH = "trained_vaes/VAELow_new.pt"
state_dict_vae = torch.load(PATH)['state_dict']
encoder.load_state_dict(state_dict_vae)

######################LOAD GRU####################################
RNN = GRUNet(seq_in=16, seq_out=1, n_layers=1, drop_prob=0.0, input_action_dim=3, input_states_dim=32, output_dim=32)

optimizer = optim.Adam(RNN.parameters(), lr=0.001)

scheduler = StepLR(optimizer, step_size=3, gamma=0.8)

#####################TRAIN GRU########################
nb_epochs = 20
for epoch in range(1, nb_epochs + 1):
    train_gru(RNN, encoder, optimizer, epoch, train_IRdata_loaded)
    scheduler.step()
######################EVAL GRU###################################
"""RNN = GRUNet(seq_in=16, seq_out=1, n_layers=1, drop_prob=0.0, input_action_dim=3, input_states_dim=32, output_dim=32)
PATH = "rnn/GRU_seqin16_seqout5_1layer_45epochs_60000steps_MultiStep2_Soutenance_SCALE13E-1_batch128_Adam_lr5E-3VAL17.pt"
state_dict_gru = torch.load(PATH)['state_dict']
RNN.load_state_dict(state_dict_gru)
"""
val_gru(RNN, encoder, eval_IRdata_loaded)

"""RNN = GRUNet(seq_in=16, seq_out=1, n_layers=1, drop_prob=0.0, input_action_dim=3, input_states_dim=32, output_dim=32)
PATH = "rnn/GRU_seqin16_seqout3_1layer_45epochs_60000steps_MultiStep2_Soutenance_SCALE13E-1_batch128_Adam_lr5E-3VAL17.pt"
state_dict_gru = torch.load(PATH)['state_dict']
RNN.load_state_dict(state_dict_gru)"""

"""plt.plot(gru.losses_training)"""
if SAVE_GRU:
    name = "myGRU_new"
    torch.save({'state_dict': RNN.state_dict()}, "trained_rnns/"+name+".pt")
 #   plt.savefig("trained_rnns/training_loss/"+name)
#plt.show()