from algos import generate_psi
from simulation_utils import create_env
import numpy as np
import sys
import os

task = sys.argv[1].lower()
K = int(sys.argv[2])

simulation_object = create_env(task)
z = simulation_object.feed_size
lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
upper_input_bound = [x[1] for x in simulation_object.feed_bounds]
inputs_set = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(K, 2*z))
psi_set = generate_psi(simulation_object, inputs_set)

if not os.path.isdir('ctrl_samples'):
    os.mkdir('ctrl_samples')
np.savez('ctrl_samples/' + simulation_object.name + '.npz', inputs_set=inputs_set, psi_set=psi_set)
print('Done!')
