from sampling import Sampler
import algos
import numpy as np
from simulation_utils import create_env, get_feedback, run_algo
import sys

def batch(task, method, N, M, b):
    if N % b != 0:
        print('N must be divisible to b')
        exit(0)
    B = 20*b

    simulation_object = create_env(task)
    d = simulation_object.num_of_features
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    psi_set = []
    s_set = []
    inputA_set = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(b, 2*simulation_object.feed_size))
    inputB_set = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(b, 2*simulation_object.feed_size))
    for j in range(b):
        input_A = inputA_set[j]
        input_B = inputB_set[j]
        psi, s = get_feedback(simulation_object, input_A, input_B)
        psi_set.append(psi)
        s_set.append(s)
    i = b
    while i < N:
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
        mean_w_samples = np.mean(w_samples,axis=0)
        print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
        print('Samples so far: ' + str(i))
        inputA_set, inputB_set = run_algo(method, simulation_object, w_samples, b, B)
        for j in range(b):
            input_A = inputA_set[j]
            input_B = inputB_set[j]
            psi, s = get_feedback(simulation_object, input_B, input_A)
            psi_set.append(psi)
            s_set.append(s)
        i += b
    w_sampler.A = psi_set
    w_sampler.y = np.array(s_set).reshape(-1,1)
    w_samples = w_sampler.sample(M)
    mean_w_samples = np.mean(w_samples, axis=0)
    print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))



def nonbatch(task, method, N, M):
    simulation_object = create_env(task)
    d = simulation_object.num_of_features
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    psi_set = []
    s_set = []
    input_A = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(2*simulation_object.feed_size))
    input_B = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(2*simulation_object.feed_size))
    psi, s = get_feedback(simulation_object, input_A, input_B)
    psi_set.append(psi)
    s_set.append(s)
    for i in range(1, N):
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
        mean_w_samples = np.mean(w_samples,axis=0)
        print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
        input_A, input_B = run_algo(method, simulation_object, w_samples)
        psi, s = get_feedback(simulation_object, input_A, input_B)
        psi_set.append(psi)
        s_set.append(s)
    w_sampler.A = psi_set
    w_sampler.y = np.array(s_set).reshape(-1,1)
    w_samples = w_sampler.sample(M)
    print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))


