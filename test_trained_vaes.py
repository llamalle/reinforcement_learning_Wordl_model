import sys
sys.path.insert(1, '/home/louis/projU2IS/controller/vae')
sys.path.insert(1, '/home/louis/projU2IS/controller/trained_vaes')
sys.path.insert(1, '/home/louis/projU2IS/controller/data')
from dataset_structures import DataRecord
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import torch
from torch.utils.data import DataLoader
from vaes import VAELow, val_vae, train_vae, loss_vae


NB_DISPLAY = 5
SAVE_DISPLAY = False
EVAL_VAE = False

agent_name = 'robot'
sensor_name = 'IR_1'

eval_data = DataRecord(
    "data/new_recordings.p", train=False, train_rate=0.8,
    step_sample=1, step_dot=1)
eval_loaded = DataLoader(eval_data, batch_size=64, shuffle=True)

mini = eval_data.minimaxis['states'][agent_name][sensor_name][0]
maxi = eval_data.minimaxis['states'][agent_name][sensor_name][1]

name = "trained_vaes/VAELow_new.pt"
encoder = VAELow()
state_dict_VAE = torch.load(name)['state_dict']
encoder.load_state_dict(state_dict_VAE)

if EVAL_VAE:
    val_vae(encoder, eval_loaded, agent_name, sensor_name)

random_indexes = [randint(0, len(eval_loaded.dataset)-1) for i in range(NB_DISPLAY)]
theta = np.linspace(0, -256/360*2*np.pi, encoder.dim_input_states)

with torch.no_grad():
    for index in random_indexes:
        print("####### Sample n"+str(index)+" #######")
        sample = eval_data[index]
        observation = sample['states'][agent_name][sensor_name]
        recons_observation, moy, logvar = encoder(torch.Tensor(observation))
        recons_observation = recons_observation.detach()
        loss = loss_vae(recons_observation[0], torch.Tensor(observation), moy, logvar).item()

        observation = observation*(maxi-mini)+mini
        recons_observation = recons_observation*(maxi-mini)+mini

        X_true = observation * np.cos(theta)
        Y_true = observation * np.sin(theta)
        X_recons = np.array(recons_observation[0]) * np.cos(theta)
        Y_recons = np.array(recons_observation[0]) * np.sin(theta)

        plt.scatter(X_true, Y_true, s=0.1, color='darkgreen', marker='o', label='$o_{t}$')
        plt.scatter(X_recons, Y_recons, s=0.1, color='fuchsia', marker='x', label='$ô_{t}$')
        plt.scatter(0, 0, color='red')
        plt.axis('equal')
        plt.legend(markerscale=15, fontsize=20, handletextpad=0.1)
        plt.title('$L_2(o_t,ô_t)=$'+str(round(loss, 3)), fontsize=20)
        plt.show()


        latent = encoder.reparametrize(moy, logvar)
        plt.plot(latent[0].tolist(), linewidth=2, color='red')
        plt.title('$Observation\ condensée$', fontsize=20)
        plt.show()
"""
       if SAVE_DISPLAY:
            plt.savefig("images/"+name+"_on_sample_"+str(index), dpi=500)
        plt.show()"""

